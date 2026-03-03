function [X_est_history, P_history] = alg_fkf(SimData)
% =========================================================================
% alg_fkf
% 联邦卡尔曼滤波 (FKF) 鲁棒算法封装
% 基于论文《Fractional Kalman filters》Algorithm 1 修正
% =========================================================================

% 1. 解包数据
GPS_Meas = SimData.GPS_Meas;
Radar_Meas = SimData.Radar_Meas;
Radar_Pos = SimData.Pos_Radar;
TotalSteps = length(SimData.Time);
outlier_ranges = SimData.outlier_ranges;
R_true_hist = SimData.R_true_hist;
Time = SimData.Time;

% 2. 滤波器初始化
X0 = zeros(6,1);
P0 = diag([100, 100, 100, 1, 1, 1]); 
P0 = diag([1e4, 1e4, 1e4, 100, 100, 100]);

% 过程噪声
Q_std_true = SimData.Q_bias_std;
% 注意：修正后的算法会在内部通过 1/Delta 自动放大 Q，
% 所以这里的 tuning_factor 可以先保持为 1，或者根据需要微调。
tuning_factor = 1;  
Q = diag(Q_std_true.^2) * tuning_factor;
F_rw = eye(6); 

% 计算 Fixed R (与 Standard SRCKF 保持一致，作为 FKF 的基准输入)
is_normal_mask = true(1, TotalSteps);
for k = 1:TotalSteps
    t = Time(k);
    for w = 1:size(outlier_ranges, 1)
        if t >= outlier_ranges(w,1) && t <= outlier_ranges(w,2)
            is_normal_mask(k) = false; 
            break;
        end
    end
end
R_nominal = mean(R_true_hist(:, is_normal_mask), 2); 
R_fix = diag(R_nominal)*10; 

% --- FKF 特有参数 ---
N_fkf = 20;           % 膨胀系数/分支数
fkf_type = 'variable'; % 变步长策略

% 状态变量 (FKF 使用 P 阵)
X_fkf = X0;
P_fkf = P0;

% 3. 滤波循环
X_est_history = zeros(6, TotalSteps);
P_history = zeros(6, 6, TotalSteps);

% 状态转移函数
f_func = @(x) F_rw * x;

for k = 1:TotalSteps
    Z_k = Radar_Meas(:, k);
    meas_func = @(x) h_meas_bias(x, GPS_Meas(1:3,k), GPS_Meas(4:6,k), Radar_Pos);
    
    % 执行一步 FKF (调用修正后的核心函数)
    [X_fkf, P_fkf] = run_fkf_step_local(X_fkf, P_fkf, Z_k, Q, R_fix, f_func, meas_func, N_fkf, fkf_type);
    
    X_est_history(:, k) = X_fkf;
    P_history(:, :, k) = P_fkf;
end

end

%% === 内部辅助函数 (FKF Core - 修正版) ===

function [x_post, P_post] = run_fkf_step_local(x_p, P_p, z, Q, R, f_func, h_func, N, type)
    n = length(x_p);
    
    % 1. 设置权重 (Fractional Exponents)
    % Delta用于量测噪声 R 的分割，Bar_Delta 用于过程噪声 Q 的分割
    if strcmp(type, 'variable')
        idx = 1:N;
        Delta = 1 ./ (idx .* (idx + 1));
        Delta = Delta / sum(Delta);
        
        % 根据论文和参考代码，过程噪声通常使用均匀分配 (Case 2)
        Bar_Delta = repmat(1/N, 1, N); 
    else
        Delta = repmat(1/N, 1, N);
        Bar_Delta = repmat(1/N, 1, N);
    end
    
    % 初始化融合累加器 (Information Fusion Accumulators)
    P_inv_sum = zeros(n, n);
    Px_inv_sum = zeros(n, 1);
    
    % 2. 并行处理 (预测 + 更新 都在分支内独立进行)
    % 修正点：移除了原来的"统一预测"，改为在循环内独立预测
    for i = 1:N
        delta_i = Delta(i);
        bar_delta_i = Bar_Delta(i);
        
        % --- A. 独立预测 (Independent Prediction) ---
        % 修正关键：Q 被放大，但上一时刻的 P_p 保持完整（不除以 delta）
        Qi = Q / bar_delta_i; 
        
        % 使用上一时刻 *完整* 的 x_p, P_p 进行 Sigma 点采样
        [Xi, W] = get_cubature_points_P(x_p, P_p);
        x_pred_pts = zeros(n, 2*n);
        for j = 1:2*n
            x_pred_pts(:,j) = f_func(Xi(:,j));
        end
        x_pred_i = x_pred_pts * W';
        
        % 计算预测协方差 (加上放大的 Qi)
        P_pred_i = Qi; 
        for j = 1:2*n
            err = x_pred_pts(:,j) - x_pred_i;
            P_pred_i = P_pred_i + W(j) * (err * err');
        end
        
        % --- B. 分数测量更新 (Fractional Measurement Update) ---
        % 测量噪声放大 (R 除以 delta)
        Ri = R / delta_i;
        
        % 基于当前分支的预测值进行采样
        [Xi_m, W_m] = get_cubature_points_P(x_pred_i, P_pred_i);
        z_pts = zeros(length(z), 2*n);
        for j = 1:2*n
            z_pts(:,j) = h_func(Xi_m(:,j));
        end
        z_pred_i = z_pts * W_m';
        
        % 计算新息协方差
        Pzz = Ri; % 包含放大的测量噪声
        Pxz = zeros(n, length(z));
        for j = 1:2*n
            res_z = z_pts(:,j) - z_pred_i;
            res_x = Xi_m(:,j) - x_pred_i;
            Pzz = Pzz + W_m(j) * (res_z * res_z');
            Pxz = Pxz + W_m(j) * (res_x * res_z');
        end
        
        % 卡尔曼更新
        K = Pxz / Pzz;
        x_upd = x_pred_i + K * (z - z_pred_i);
        P_upd = P_pred_i - K * Pzz * K';
        
        % --- C. 融合累加 (Information Form) ---
        % 使用 pinv 增强数值稳定性
        invP = pinv(P_upd); 
        P_inv_sum = P_inv_sum + invP;
        Px_inv_sum = Px_inv_sum + invP * x_upd;
    end
    
    % 3. 最终融合结果 (Fractional State Fusion)
    P_post = pinv(P_inv_sum);
    x_post = P_post * Px_inv_sum;
    P_post = (P_post + P_post') / 2; % 强制对称
end

function [Xi, W] = get_cubature_points_P(x, P)
    n = length(x);
    nPts = 2*n;
    W = repmat(1/nPts, 1, nPts);
    try
        S = chol(P, 'lower');
    catch
        [V, D] = eig(P);
        S = chol(V * max(D, 1e-8) * V', 'lower');
    end
    Xi = repmat(x, 1, nPts) + S * sqrt(n) * [eye(n), -eye(n)];
end

function z = h_meas_bias(bias_state, gps_pos, gps_vel, radar_pos)
    pos_est = gps_pos - bias_state(1:3);
    vel_est = gps_vel - bias_state(4:6);
    diff = pos_est - radar_pos;
    dist = norm(diff);
    if dist < 1e-3, dist = 1e-3; end
    range_rate = dot(diff, vel_est) / dist;
    z = [dist; range_rate];
end