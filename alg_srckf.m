function [X_est_history, P_history] = alg_srckf(SimData)
% =========================================================================
% alg_srckf
% 标准 SRCKF 算法封装 (用于估计 GPS Bias)
% =========================================================================

% 1. 解包数据
GPS_Meas = SimData.GPS_Meas;
Radar_Meas = SimData.Radar_Meas;
Radar_Pos = SimData.Pos_Radar;
dt = SimData.dt;
TotalSteps = length(SimData.Time);

% 2. 滤波器初始化 (保持 main3.m 参数)
% 状态: [Bias_Pos; Bias_Vel] (6x1)
X0 = zeros(6,1);
P0 = diag([100, 100, 100, 1, 1, 1]); 
S0 = chol(P0, 'lower');

% 过程噪声 (Bias 是随机游走)
Q_std_true = SimData.Q_bias_std;
tuning_factor = 1;  % 尝试 10, 50, 100
Q = diag(Q_std_true.^2) * tuning_factor;
SQ = chol(Q, 'lower');
F_rw = eye(6); % 随机游走矩阵

% 量测噪声 R_fix 策略 (保持 main3.m 逻辑: 非粗差段平均值 * 10)
outlier_ranges = SimData.outlier_ranges;
R_true_hist = SimData.R_true_hist;
Time = SimData.Time;
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
% R_fix = diag(R_nominal);

% 3. 滤波循环
X_est_history = zeros(6, TotalSteps); % 存储估计的 Bias
P_history = zeros(6, 6, TotalSteps);  % 可选：存储协方差

X_curr = X0;
S_curr = S0;

% 定义状态转移函数 (线性)
f_func = @(x) F_rw * x;

for k = 1:TotalSteps
    Z_k = Radar_Meas(:, k);
    
    % 定义观测函数 (闭包: 需要当前的 GPS 读数)
    % 注意：这里估计的是 Bias，所以观测函数是 h(Bias, GPS) -> RadarRange
    meas_func = @(x) h_meas_bias(x, GPS_Meas(1:3,k), GPS_Meas(4:6,k), Radar_Pos);
    
    % 执行一步 SRCKF
    [X_curr, S_curr] = run_srckf_step_local(X_curr, S_curr, F_rw, SQ, Z_k, R_fix, meas_func);
    
    % 存储结果 (Bias)
    X_est_history(:, k) = X_curr;
    P_history(:, :, k) = S_curr * S_curr';
end

end

%% === 内部辅助函数 (源自 main3.m) ===

function [X_new, S_new] = run_srckf_step_local(X, S, F, SQ, Z, R, h_fun)
    [X_p, S_p] = srckf_predict(X, S, F, SQ);
    [X_new, S_new, ~] = srckf_update(X_p, S_p, Z, R, h_fun);
end

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

function [X_new, S_new, Pzz_noiseless] = srckf_update(X_pred, S_pred, Z, R, h_fun)
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
    try 
        S_new = chol(P_new, 'lower'); 
    catch
        S_new = chol(P_new + 1e-6*eye(n), 'lower'); 
    end
end

function Pts = cubature_pts(x, S)
    n = length(x);
    Xi = sqrt(n) * [eye(n), -eye(n)];
    Pts = x + S * Xi;
end

% 观测函数：h(bias) = Range(GPS - bias, Radar)
function z = h_meas_bias(bias_state, gps_pos, gps_vel, radar_pos)
    pos_est = gps_pos - bias_state(1:3);
    vel_est = gps_vel - bias_state(4:6);
    diff = pos_est - radar_pos;
    dist = norm(diff);
    if dist < 1e-3, dist = 1e-3; end
    range_rate = dot(diff, vel_est) / dist;
    z = [dist; range_rate];
end