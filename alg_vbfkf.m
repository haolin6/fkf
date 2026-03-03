function [X_est_history, P_history] = alg_vbfkf(SimData)
% =========================================================================
% alg_vbfkf
% 变分贝叶斯联邦卡尔曼滤波 (VB-FKF) - 修正增强版
% 核心特征：
% 1. 结构：基于 FKF (N=20, 变步长) 抗非线性/粗差。
% 2. 自适应：外层包裹 VB 循环，在线估计噪声协方差 R。
% 3. 改进：引入自适应权重分配 (Adaptive Weighting) 并增加稳定性控制。
% =========================================================================

% 1. 解包数据
GPS_Meas = SimData.GPS_Meas;
Radar_Meas = SimData.Radar_Meas;
Radar_Pos = SimData.Pos_Radar;
dt = SimData.dt;
TotalSteps = length(SimData.Time);
outlier_ranges = SimData.outlier_ranges; % 仅用于辅助逻辑(如果需要)
R_true_hist = SimData.R_true_hist;
Time = SimData.Time;

% 2. 滤波器初始化
X0 = zeros(6,1);
P0 = diag([100, 100, 100, 1, 1, 1]); 
S0 = chol(P0, 'lower'); 


% 计算初始 R (R_fix) 用于 VB 初始化
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

% 过程噪声
Q_std_true = SimData.Q_bias_std;
tuning_factor = 10;  % 尝试 10, 50, 100
Q = diag(Q_std_true.^2) * tuning_factor;
SQ = chol(Q, 'lower');
F_rw = eye(6); 

% --- VB 参数初始化 ---
% 初始猜测: 给一个较大的初值，让 VB 自行收敛
% R_init = diag([100, 10]); 
n_meas = 2;
rho = 0.9; % 遗忘因子

% Inverse-Wishart 分布参数
u_vb_0 = n_meas + 2; 
U_vb_0 = R_fix * (u_vb_0 - n_meas - 1); 

u_k = u_vb_0; U_k = U_vb_0; 

% --- FKF 参数 ---
N_fkf = 20;            % FKF 分支数
fkf_type = 'variable'; % 变步长策略

% 状态变量
X_curr = X0; 
P_curr = P0; 

% 【新增】定义稳定参考 R 和下限 R_min
% 参考 R：用于计算粗差判决统计量 (不要用动态的 R_est，否则不稳定)
R_ref = diag([25, 1]); % 距离方差25，速度方差1 (基于名义噪声)
% 下限 R：防止 VB 过拟合导致 R 接近 0
R_min_limit = diag([1^2, 0.1^2]); 

% 3. 滤波循环
X_est_history = zeros(6, TotalSteps);
P_history = zeros(6, 6, TotalSteps);

% 状态转移函数句柄
f_func = @(x) F_rw * x;

for k = 1:TotalSteps
    Z_k = Radar_Meas(:, k);
    % 观测函数 (闭包)
    meas_func = @(x) h_meas_bias(x, GPS_Meas(1:3,k), GPS_Meas(4:6,k), Radar_Pos);
    
    % --- 步骤 1: 公共预测 (Common Prediction) ---
    [X_pred, P_pred] = ckf_prediction(X_curr, P_curr, f_func, Q);
    
    % --- 步骤 2: VB 先验传播 ---
    u_prior = rho * (u_k - n_meas - 1) + n_meas + 1;
    U_prior = rho * U_k;
    
    % --- 步骤 3: VB 迭代 (包含 FKF Update) ---
    X_iter = X_pred; 
    P_iter = P_pred;
    u_loop = u_prior; 
    U_loop = U_prior;
    
    Max_Iter = 5; % VB 迭代次数
    
    for iter = 1:Max_Iter
        % A. 估计当前的 R
        R_est = U_loop / (u_loop - n_meas - 1);
        R_est = (R_est + R_est')/2;
        
        % 【稳定性修正1】给 R_est 施加硬下限，防止过拟合噪声
        R_est = max(R_est, R_min_limit);
        
        % =======================================================
        % 【新增】自适应 FKF 权重调整 (Adaptive Weights)
        % =======================================================
        
        % 1. 计算判决统计量 (使用当前迭代状态)
        Z_pred_curr = meas_func(X_iter);
        res_curr = Z_k - Z_pred_curr;
        
        % 【稳定性修正2】使用固定的 R_ref 计算 Mahalanobis 距离
        % 避免使用 R_est 导致的"正反馈震荡"
        gamma_stat = res_curr' * (R_ref \ res_curr); 
        
        % 2. 构造自适应权重
        % 逻辑：gamma 小 -> 使用 Base 分布 (信任测量，高精度)
        %      gamma 大 -> 使用 Flat 分布 (鲁棒模式，抗粗差)
        
        % 基准权重 (Variable Step)
        idx = 1:N_fkf;
        Delta_base = 1 ./ (idx .* (idx + 1));
        Delta_base = Delta_base / sum(Delta_base);
        
        % 鲁棒权重 (Flat)
        Delta_flat = ones(1, N_fkf) / N_fkf;
        
        % 【稳定性修正3】使用 Sigmoid 平滑切换
        % 自由度=2，卡方分布 99% 分位点约为 9.21
        chi2_threshold = 9.0; 
        scale_factor = 2.0; % 控制切换的陡峭程度 (越大越平滑)
        
        alpha = 1 / (1 + exp(-(gamma_stat - chi2_threshold) / scale_factor));
        
        % 动态合成 Delta
        Delta_adaptive = (1 - alpha) * Delta_base + alpha * Delta_flat;
        Delta_adaptive = Delta_adaptive / sum(Delta_adaptive);
        
        % =======================================================

        % B. 核心融合：FKF Update (传入动态权值 Delta_adaptive)
        [X_next, P_next] = run_fkf_update_only(X_pred, P_pred, Z_k, R_est, meas_func, N_fkf, Delta_adaptive);
        
        % C. 计算 VB 所需的统计量 (用于更新 Wishart 参数)
        [Xi_post, W_post] = get_cubature_points_P(X_next, P_next);
        Z_pts_post = zeros(n_meas, size(Xi_post,2));
        for j = 1:size(Xi_post,2)
            Z_pts_post(:,j) = meas_func(Xi_post(:,j));
        end
        Z_pred_post = Z_pts_post * W_post';
        
        res_post = Z_k - Z_pred_post;
        
        % 估计 Pzz 的"非线性传播部分"
        Pzz_post_cov = zeros(n_meas, n_meas);
        for j = 1:size(Xi_post,2)
            diff_z = Z_pts_post(:,j) - Z_pred_post;
            Pzz_post_cov = Pzz_post_cov + W_post(j) * (diff_z * diff_z');
        end
        
        % 构造用于更新 Wishart 分布的矩阵 T
        T_mat = res_post * res_post' + Pzz_post_cov;
        
        % D. 更新分布参数 (u, U)
        u_loop = u_prior + 1;
        U_loop = U_prior + T_mat;
        
        % 更新迭代状态
        X_iter = X_next;
        P_iter = P_next;
    end
    
    % --- 步骤 4: 更新完成，保存状态 ---
    X_curr = X_iter;
    P_curr = P_iter;
    u_k = u_loop;
    U_k = U_loop;
    
    X_est_history(:, k) = X_curr;
    P_history(:, :, k) = P_curr;
end

end

%% === 辅助函数区域 ===

% 1. 标准 CKF 预测 (基于 P 阵)
function [x_pred, P_pred] = ckf_prediction(x_est, P_est, f_func, Q)
    n = length(x_est);
    [Xi, W] = get_cubature_points_P(x_est, P_est);
    x_pts_pred = zeros(n, 2*n);
    for i = 1:2*n
        x_pts_pred(:,i) = f_func(Xi(:,i));
    end
    x_pred = x_pts_pred * W';
    P_pred = Q;
    for i = 1:2*n
        diff = x_pts_pred(:,i) - x_pred;
        P_pred = P_pred + W(i) * (diff * diff');
    end
end

% 2. FKF 纯更新函数 (修正版：接受 Delta_in)
% 输入: 预测状态 X_p, P_p, 量测 Z, 以及 **VB估计出的 R**, **自适应权重 Delta_in**
function [x_post, P_post] = run_fkf_update_only(x_p, P_p, z, R_est, h_func, N, Delta_in)
    n = length(x_p);
    
    % 直接使用传入的自适应权重
    Delta = Delta_in; 
    
    % 初始化信息矩阵累加器
    P_inv_sum = zeros(n, n);
    Px_inv_sum = zeros(n, 1);
    
    % 并行处理 N 个 FKF 分支
    for i = 1:N
        delta_i = Delta(i);
        
        % 关键：同时放大 P (FKF特性) 和 R (VB传入的估计值)
        P_pred_i = P_p / delta_i; 
        R_i      = R_est / delta_i; 
        x_pred_i = x_p; % 均值不变
        
        % CKF 更新步骤 (针对第 i 个分支)
        [Xi_m, W_m] = get_cubature_points_P(x_pred_i, P_pred_i);
        z_pts = zeros(length(z), 2*n);
        for j = 1:2*n
            z_pts(:,j) = h_func(Xi_m(:,j));
        end
        z_pred_i = z_pts * W_m';
        
        Pzz = R_i; % 包含测量噪声
        Pxz = zeros(n, length(z));
        for j = 1:2*n
            res_z = z_pts(:,j) - z_pred_i;
            res_x = Xi_m(:,j) - x_pred_i;
            Pzz = Pzz + W_m(j) * (res_z * res_z');
            Pxz = Pxz + W_m(j) * (res_x * res_z');
        end
        
        K = Pxz / Pzz;
        x_upd = x_pred_i + K * (z - z_pred_i);
        P_upd = P_pred_i - K * Pzz * K';
        
        % 信息融合 (Information Fusion)
        % 使用 pinv 增强数值稳定性
        invP = pinv(P_upd); 
        P_inv_sum = P_inv_sum + invP;
        Px_inv_sum = Px_inv_sum + invP * x_upd;
    end
    
    % 最终融合结果
    P_post = pinv(P_inv_sum);
    x_post = P_post * Px_inv_sum;
    P_post = (P_post + P_post') / 2; % 强制对称
end

% 3. 生成 Sigma 点 (基于 P 阵)
function [Xi, W] = get_cubature_points_P(x, P)
    n = length(x);
    nPts = 2*n;
    W = repmat(1/nPts, 1, nPts);
    try
        S = chol(P, 'lower');
    catch
        % 如果 P 非正定，使用特征值分解修复
        [V, D] = eig(P);
        S = chol(V * max(D, 1e-8) * V', 'lower');
    end
    Xi = repmat(x, 1, nPts) + S * sqrt(n) * [eye(n), -eye(n)];
end

% 4. 量测函数 (Bias Model)
function z = h_meas_bias(bias_state, gps_pos, gps_vel, radar_pos)
    pos_est = gps_pos - bias_state(1:3);
    vel_est = gps_vel - bias_state(4:6);
    diff = pos_est - radar_pos;
    dist = norm(diff);
    if dist < 1e-3, dist = 1e-3; end
    range_rate = dot(diff, vel_est) / dist;
    z = [dist; range_rate];
end