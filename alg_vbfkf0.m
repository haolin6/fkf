function [X_est_history, P_history] = alg_vbfkf(SimData)
% =========================================================================
% alg_vbfkf
% 变分贝叶斯联邦卡尔曼滤波 (VB-FKF) - 实现文档中的 "方法 3"
% 核心特征：
% 1. 结构：基于 FKF (N=20, 变步长) 抗非线性/粗差。
% 2. 自适应：外层包裹 VB 循环，在线估计噪声协方差 R。
% =========================================================================

% 1. 解包数据
GPS_Meas = SimData.GPS_Meas;
Radar_Meas = SimData.Radar_Meas;
Radar_Pos = SimData.Pos_Radar;
dt = SimData.dt;
TotalSteps = length(SimData.Time);
% 注意：VB-FKF 不需要 R_true_hist 或 outlier_ranges 来设定 R_fix，
% 因为它会自己估计 R。但我们需要一个初始猜测。
outlier_ranges = SimData.outlier_ranges; % 仅用于辅助逻辑(如果需要)

% 2. 滤波器初始化
X0 = zeros(6,1);
P0 = diag([100, 100, 100, 1, 1, 1]); 
S0 = chol(P0, 'lower'); % 虽然 FKF 主要用 P，但预测步我们可以借用 CKF 的 S

% 过程噪声
Q_std_true = SimData.Q_bias_std;
Q = diag(Q_std_true.^2);
SQ = chol(Q, 'lower');
F_rw = eye(6); 

% --- VB 参数初始化 ---
% 我们给一个相对保守的初始 R (比如基于名义噪声)
R_init = diag([25, 1]); % 初始猜测: 距离方差25, 多普勒方差1
R_init = diag([100, 10]); % 给一个更大的初值
n_meas = 2;
rho = 0.9; % 遗忘因子 (控制 R 的适应速度)

% Inverse-Wishart 分布参数
u_vb_0 = n_meas + 2; 
U_vb_0 = R_init * (u_vb_0 - n_meas - 1); 

u_k = u_vb_0; U_k = U_vb_0; 

% --- FKF 参数 ---
N_fkf = 20;            % FKF 分支数/膨胀系数
fkf_type = 'variable'; % 变步长策略

% 状态变量
X_curr = X0; 
P_curr = P0; 

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
    % FKF 所有分支共享同一个先验，这里使用标准的 CKF 预测方法
    [X_pred, P_pred] = ckf_prediction(X_curr, P_curr, f_func, Q);
    
    % --- 步骤 2: VB 先验传播 ---
    u_prior = rho * (u_k - n_meas - 1) + n_meas + 1;
    U_prior = rho * U_k;
    
    % --- 步骤 3: VB 迭代 (包含 FKF Update) ---
    % 初始化迭代变量
    X_iter = X_pred; 
    P_iter = P_pred;
    u_loop = u_prior; 
    U_loop = U_prior;
    
    Max_Iter = 5; % VB 迭代次数
    
    for iter = 1:Max_Iter
        % A. 估计当前的 R
        % R_est = E[R] = U / (u - m - 1)
        R_est = U_loop / (u_loop - n_meas - 1);
        
        % 保证 R 正定 (数值稳定性)
        R_est = (R_est + R_est')/2;
        
        % B. 核心融合：FKF Update (使用估计的 R_est)
        % 这就是方法3的精髓：在更新步骤内部，利用 FKF 的结构处理非线性，
        % 但使用 VB 估计出的 R_est
        [X_next, P_next] = run_fkf_update_only(X_pred, P_pred, Z_k, R_est, meas_func, N_fkf, fkf_type);
        
        % C. 计算 VB 所需的统计量
        % 我们需要计算残差协方差矩阵 F_mat 或者是 Pzz_post
        % 这里使用后验状态重新计算残差 (或者可以使用 FKF 融合过程中的副产品，
        % 但为了代码清晰，我们用后验状态重算一次 sigma 点)
        
        % 生成基于后验的 sigma 点
        [Xi_post, W_post] = get_cubature_points_P(X_next, P_next);
        Z_pts_post = zeros(n_meas, size(Xi_post,2));
        for j = 1:size(Xi_post,2)
            Z_pts_post(:,j) = meas_func(Xi_post(:,j));
        end
        Z_pred_post = Z_pts_post * W_post';
        
        % 计算后验残差
        res_post = Z_k - Z_pred_post;
        
        % 估计 Pzz 的"噪声部分" (近似)
        % 在标准 VB-KF 中，N = (z-Hx)(z-Hx)' + HPH'。
        % 这里是非线性，我们可以近似认为:
        % E[ (z-h(x))(z-h(x))' ] ≈ res * res' + H_eff * P_post * H_eff' 
        % 利用 Sigma 点计算 output covariance:
        Pzz_post_cov = zeros(n_meas, n_meas);
        for j = 1:size(Xi_post,2)
            diff_z = Z_pts_post(:,j) - Z_pred_post;
            Pzz_post_cov = Pzz_post_cov + W_post(j) * (diff_z * diff_z');
        end
        
        % 构造用于更新 Wishart 分布的矩阵 T (或 F_mat)
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

% 2. FKF 纯更新函数 (The Core of Method 3)
% 输入: 预测状态 X_p, P_p, 量测 Z, 以及 **VB估计出的 R**
function [x_post, P_post] = run_fkf_update_only(x_p, P_p, z, R_est, h_func, N, type)
    n = length(x_p);
    
    % 设置权重 (Variable Step)
    if strcmp(type, 'variable')
        idx = 1:N;
        Delta = 1 ./ (idx .* (idx + 1));
        Delta = Delta / sum(Delta);
    else
        Delta = repmat(1/N, 1, N);
    end
    
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