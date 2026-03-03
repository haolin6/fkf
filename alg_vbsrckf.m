function [X_est_history, P_history] = alg_vbsrckf(SimData)
% =========================================================================
% alg_vbsrckf
% 变分贝叶斯 (VB-SRCKF) 算法封装
% =========================================================================

% 1. 解包数据
GPS_Meas = SimData.GPS_Meas;
Radar_Meas = SimData.Radar_Meas;
Radar_Pos = SimData.Pos_Radar;
dt = SimData.dt;
TotalSteps = length(SimData.Time);
outlier_ranges = SimData.outlier_ranges;
R_true_hist = SimData.R_true_hist;
Time = SimData.Time;

% 2. 滤波器初始化
X0 = zeros(6,1);
P0 = diag([100, 100, 100, 1, 1, 1]); 
S0 = chol(P0, 'lower');

% 过程噪声
Q_std_true = SimData.Q_bias_std;
Q = diag(Q_std_true.^2);
SQ = chol(Q, 'lower');
F_rw = eye(6); 

% --- VB 特定参数 (源自 main3.m) ---
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

% VB 超参数
rho = 0.9; 
n_meas = 2;
u_vb_0 = n_meas + 2; 
U_vb_0 = R_fix * (u_vb_0 - n_meas - 1); 
a_vb_0 = 1; b_vb_0 = 1;

% 状态变量
X_vb = X0; S_vb = S0; 
u_k = u_vb_0; U_k = U_vb_0; 
a_k = a_vb_0; b_k = b_vb_0;

% 3. 滤波循环
X_est_history = zeros(6, TotalSteps);
P_history = zeros(6, 6, TotalSteps);

for k = 1:TotalSteps
    Z_k = Radar_Meas(:, k);
    meas_func = @(x) h_meas_bias(x, GPS_Meas(1:3,k), GPS_Meas(4:6,k), Radar_Pos);
    
    % --- VB-SRCKF Step ---
    % 1. 预测
    [X_pred, S_pred] = srckf_predict(X_vb, S_vb, F_rw, SQ);
    
    % 2. VB 参数先验传播
    u_prior = rho * (u_k - n_meas - 1) + n_meas + 1;
    U_prior = rho * U_k;
    a_prior = rho * (a_k - 1) + 1;
    b_prior = rho * b_k;
    
    % 3. VB 迭代更新 (Fixed 5 iterations)
    X_iter = X_pred; S_iter = S_pred;
    u_loop = u_prior; U_loop = U_prior; a_loop = a_prior; b_loop = b_prior;
    
    for iter = 1:5
        % 估计当前的 R
        R_est = inv(u_loop * inv(U_loop)) * (b_loop / a_loop);
        
        % 使用当前 R 进行量测更新
        [X_next, S_next, Pzz_noise] = srckf_update_vb(X_pred, S_pred, Z_k, R_est, meas_func);
        
        % 计算残差统计量 F_mat
        Z_pred_post = meas_func(X_next);
        res = Z_k - Z_pred_post;
        F_mat = res * res' + Pzz_noise;
        
        % 更新分布参数
        u_loop = u_prior + 1;
        U_loop = U_prior + (a_loop/b_loop) * F_mat;
        a_loop = a_prior + 0.5 * n_meas;
        b_loop = b_prior + 0.5 * trace(F_mat * (u_loop * inv(U_loop)));
        
        X_iter = X_next; S_iter = S_next;
    end
    
    % 保存状态
    X_vb = X_iter; S_vb = S_iter;
    u_k = u_loop; U_k = U_loop; a_k = a_loop; b_k = b_loop;
    
    X_est_history(:, k) = X_vb;
    P_history(:, :, k) = S_vb * S_vb';
end

end

%% === 内部辅助函数 ===

function [X_pred, S_pred] = srckf_predict(X_est, S_est, F, SQ)
    n = length(X_est);
    nPts = 2*n;
    Pts = cubature_pts(X_est, S_est);
    X_star = F * Pts;
    X_pred = mean(X_star, 2);
    Weighted_Err = (X_star - X_pred) / sqrt(nPts);
    [~, R_qr] = qr([Weighted_Err, SQ]', 0);
    S_pred = R_qr';
end

function [X_new, S_new, Pzz_noiseless] = srckf_update_vb(X_pred, S_pred, Z, R, h_fun)
    n = length(X_pred);
    nPts = 2*n;
    Pts = cubature_pts(X_pred, S_pred);
    m = length(Z);
    Z_star = zeros(m, nPts);
    for i = 1:nPts
        Z_star(:, i) = h_fun(Pts(:, i));
    end
    Z_pred = mean(Z_star, 2);
    Weighted_Z = (Z_star - Z_pred) / sqrt(nPts);
    Weighted_X = (Pts - X_pred) / sqrt(nPts);
    Pzz_noiseless = Weighted_Z * Weighted_Z';
    Pzz = Pzz_noiseless + R;
    Pxz = Weighted_X * Weighted_Z';
    K = Pxz / Pzz;
    Inn = Z - Z_pred;
    X_new = X_pred + K * Inn;
    P_new = S_pred*S_pred' - K * Pzz * K';
    P_new = (P_new + P_new')/2;
    try S_new = chol(P_new, 'lower'); catch, S_new = chol(P_new + 1e-6*eye(n), 'lower'); end
end

function Pts = cubature_pts(x, S)
    n = length(x);
    Xi = sqrt(n) * [eye(n), -eye(n)];
    Pts = x + S * Xi;
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