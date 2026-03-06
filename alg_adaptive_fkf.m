function [X_est_history, P_history] = alg_adaptive_fkf(SimData)

GPS_Meas   = SimData.GPS_Meas;
Radar_Meas = SimData.Radar_Meas;
Radar_Pos  = SimData.Pos_Radar;
TotalSteps = length(SimData.Time);
R_true_hist = SimData.R_true_hist;

% ========= 固定 GPS 噪声 =========
sigma_gps_pos = 2;
sigma_gps_vel = 0.1;
% ==================================

X0 = zeros(6,1);
P0 = diag([100,100,100,1,1,1]);

tuning_factor = 1;
Q = diag(SimData.Q_bias_std.^2)*tuning_factor;
F_rw = eye(6);

R_nominal = mean(R_true_hist,2);
R_fix = diag(R_nominal);

N_max = 5;
fkf_type = 'variable';

chi2_threshold = 5;

X_fkf = X0;
P_fkf = P0;

X_est_history = zeros(6,TotalSteps);
P_history     = zeros(6,6,TotalSteps);

f_func = @(x) F_rw*x;

for k = 1:TotalSteps

    Z_k = Radar_Meas(:,k);
    gps_pos = GPS_Meas(1:3,k);
    gps_vel = GPS_Meas(4:6,k);

    meas_func = @(x) h_meas_bias(x,gps_pos,gps_vel,Radar_Pos);

    [X_fkf,P_fkf,current_N] = run_adaptive_fkf_step( ...
        X_fkf,P_fkf,Z_k,Q,R_fix,...
        f_func,meas_func,...
        gps_pos,gps_vel,Radar_Pos,...
        sigma_gps_pos,sigma_gps_vel,...
        N_max,fkf_type,chi2_threshold);

    X_est_history(:,k) = X_fkf;
    P_history(:,:,k)   = P_fkf;

    if current_N > 1
        fprintf('Time %d: Jump detected! Adaptive N = %d\n',k,current_N);
    end
end

end
function [x_post,P_post,N_adapt] = run_adaptive_fkf_step( ...
    x_p,P_p,z,Q,R_radar,...
    f_func,h_func,...
    gps_pos,gps_vel,radar_pos,...
    sigma_gps_pos,sigma_gps_vel,...
    N_max,type,chi2_th)

n = length(x_p);

%% 公共预测
[Xi,W] = get_cubature_points_P(x_p,P_p);

x_pred_pts = zeros(n,2*n);
for j=1:2*n
    x_pred_pts(:,j) = f_func(Xi(:,j));
end

x_pred_common = x_pred_pts*W';

P_pred_common = Q;
for j=1:2*n
    err = x_pred_pts(:,j)-x_pred_common;
    P_pred_common = P_pred_common + W(j)*(err*err');
end

%% ================= Test 步骤 =================

[Xi_test,W_test] = get_cubature_points_P(x_pred_common,P_pred_common);

z_pts_test = zeros(length(z),2*n);
for j=1:2*n
    z_pts_test(:,j) = h_func(Xi_test(:,j));
end

z_pred_test = z_pts_test*W_test';

Pzz_noiseless = zeros(length(z));
for j=1:2*n
    res_z = z_pts_test(:,j)-z_pred_test;
    Pzz_noiseless = Pzz_noiseless + W_test(j)*(res_z*res_z');
end

% ===== 计算 R_eff (Test 阶段) =====
bias_pos = x_pred_common(1:3);
bias_vel = x_pred_common(4:6);

pos_est = gps_pos - bias_pos;
vel_est = gps_vel - bias_vel;

R_eff_test = compute_R_eff( ...
                pos_est,vel_est,radar_pos,...
                R_radar,...
                sigma_gps_pos,sigma_gps_vel);

S_test = Pzz_noiseless + R_eff_test;

gamma = z - z_pred_test;
lambda = gamma'*(S_test\gamma);

%% 自适应 N
if lambda <= chi2_th
    N_adapt = 1;
else
    N_adapt = N_max;
end
% ====== lambda -> N 渐进映射（替换原二值逻辑）======
% 直觉：lambda 只略高于阈值 -> N 小；lambda 很大 -> N 接近 N_max
% 
% lambda_hi = 6 * chi2_th;   % "很异常"的上界，可调：3~6倍 chi2_th
% if lambda <= chi2_th
%     N_adapt = 1;
% else
%     % 归一化严重程度 s in [0,1]
%     s = (lambda - chi2_th) / (lambda_hi - chi2_th);
%     s = max(0, min(1, s));
% 
%     % 映射到 [1, N_max]
%     N_adapt = 1 + floor((N_max - 1) * s);
% 
%     % 保底：至少2（如果你希望一旦触发就至少用FKF）
%     N_adapt = max(N_adapt, 3);
% end
% % ==============================================

%% ================= FKF 更新 =================

if strcmp(type,'variable') && N_adapt>1
    idx = 1:N_adapt;
    Delta = 1./(idx.*(idx+1));
    Delta = Delta/sum(Delta);
    Bar_Delta = repmat(1/N_adapt,1,N_adapt);
else
    Delta = repmat(1/N_adapt,1,N_adapt);
    Bar_Delta = repmat(1/N_adapt,1,N_adapt);
end

P_inv_sum = zeros(n);
Px_inv_sum = zeros(n,1);

for i=1:N_adapt

    P_pred_i = P_pred_common/Bar_Delta(i);
    x_pred_i = x_pred_common;

    % ===== 重新计算 R_eff =====
    bias_pos = x_pred_i(1:3);
    bias_vel = x_pred_i(4:6);

    pos_est = gps_pos - bias_pos;
    vel_est = gps_vel - bias_vel;

    R_eff = compute_R_eff( ...
                pos_est,vel_est,radar_pos,...
                R_radar,...
                sigma_gps_pos,sigma_gps_vel);

    R_i = R_eff/Delta(i);

    [Xi_m,W_m] = get_cubature_points_P(x_pred_i,P_pred_i);

    z_pts = zeros(length(z),2*n);
    for j=1:2*n
        z_pts(:,j) = h_func(Xi_m(:,j));
    end

    z_pred_i = z_pts*W_m';

    Pzz = R_i;
    Pxz = zeros(n,length(z));

    for j=1:2*n
        res_z = z_pts(:,j)-z_pred_i;
        res_x = Xi_m(:,j)-x_pred_i;
        Pzz = Pzz + W_m(j)*(res_z*res_z');
        Pxz = Pxz + W_m(j)*(res_x*res_z');
    end

    K = Pxz/Pzz;

    x_upd_i = x_pred_i + K*(z-z_pred_i);
    P_upd_i = P_pred_i - K*Pzz*K';
    P_upd_i = (P_upd_i+P_upd_i')/2;

    invP = pinv(P_upd_i);
    P_inv_sum  = P_inv_sum  + invP;
    Px_inv_sum = Px_inv_sum + invP*x_upd_i;
end

P_post = pinv(P_inv_sum);
x_post = P_post*Px_inv_sum;
P_post = (P_post+P_post')/2;

end
function R_eff = compute_R_eff(pos_est,vel_est,radar_pos,...
                               R_radar,...
                               sigma_gps_pos,sigma_gps_vel)

d = pos_est - radar_pos;
r = norm(d);
if r < 1e-6
    r = 1e-6;
end

los  = d/r;
rdot = dot(d,vel_est)/r;

J = zeros(2,6);
J(1,1:3) = los';
J(2,4:6) = los';
J(2,1:3) = vel_est'/r - (rdot/r)*los';

R_gps = diag([ ...
    sigma_gps_pos^2*ones(1,3), ...
    sigma_gps_vel^2*ones(1,3)]);

R_eff = R_radar + J*R_gps*J';
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
