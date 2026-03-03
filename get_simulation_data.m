function SimData = get_simulation_data(run_seed)
% =========================================================================
% get_simulation_data
% 基于原 scenenonlinear.m 改编
% 输入: run_seed (随机种子，用于蒙特卡洛区分每次实验)
% 输出: SimData (包含所有真值、量测和参数的结构体)
% =========================================================================

% 1. 设置随机种子
rng(run_seed); 

% 2. 基础参数 (保持原文件一致)
T = 30;                 
dt = 0.1;               
TotalSteps = T / dt;

Pos_Radar = [0; 150; 0]; % Y=50m, 强非线性
% Pos_Radar = [0; 0.1; 0]; % 原来是 [0; 50; 0]，现在改为 Y=10m
% Pos_Radar = [0; 1000; 0]; % 改为 1000m

outlier_ranges = [
    8.0,  12.0;   
    20.0, 22.0
];
outlier_ranges = [];

% 3. 生成真实轨迹 (CV模型)
X0 = -1500; 
Y0 = 0; 
H0 = 100;      
Vx = 150;      

true_state = zeros(6, TotalSteps); 
curr_state = [X0; Y0; H0; Vx; 0; 0];

q_proc = 0.5;
% q_proc = 500.0; % 原来是 0.5，现在放大 10 倍
Q_cv = [0.5*dt^2; 0.5*dt^2; 0.5*dt^2; dt; dt; dt] * q_proc; 

for k = 1:TotalSteps
    F = eye(6); F(1,4)=dt; F(2,5)=dt; F(3,6)=dt;
    
    noise_proc = randn(6,1) .* Q_cv; 
    % 保持原文件的各轴噪声比例
    noise_proc(2) = noise_proc(2) * 0.1; 
    noise_proc(3) = noise_proc(3) * 0.01; 
    noise_proc(5) = noise_proc(5) * 0.1; 
    noise_proc(6) = noise_proc(6) * 0.01; 
    
    curr_state = F * curr_state + noise_proc;
    true_state(:, k) = curr_state;
end

% 4. 生成 GPS 偏差 (Bias) - 待估计状态
Bias_true = zeros(6, TotalSteps);
curr_bias = [10; 5; -5; 0.2; 0.1; -0.1]; 
sigma_bias_change = [0.01; 0.01; 0.01; 0.001; 0.001; 0.001]*10; 

for k = 1:TotalSteps
    curr_bias = curr_bias + randn(6,1) .* sigma_bias_change;
    % 在 get_simulation_data.m 的生成真实 Bias 循环中加入：
% if k == (15 / dt) % 第10秒
%     curr_bias(1:3) = curr_bias(1:3) + 10; % 发生 20m 的突变
% end
    Bias_true(:, k) = curr_bias;
end

% 5. 生成传感器量测
% A. GPS
GPS_Meas = zeros(6, TotalSteps);
GPS_Noise_Std = diag([2, 2, 5, 0.1, 0.1, 0.1]); 

for k = 1:TotalSteps
    GPS_Meas(:, k) = true_state(:, k) + Bias_true(:, k) + GPS_Noise_Std * randn(6,1);
end

% B. Radar
Radar_Meas = zeros(2, TotalSteps);
R_true_hist = zeros(2, TotalSteps); 

for k = 1:TotalSteps
    current_time = (k-1) * dt;
    p = true_state(1:3, k);
    v = true_state(4:6, k);
    
    diff = p - Pos_Radar;
    dist = norm(diff);
    range_rate = dot(diff, v) / dist;
    
    % 距离相关噪声模型
    sig_r = 5.0 + 2.0 * (1000/dist); 
    sig_rr = 0.5 + 0.5 * (1000/dist);
    sig_r = 5.0; 
    sig_rr = 0.5;
    
    % 粗差注入
    is_outlier = false;
    for w = 1:size(outlier_ranges, 1)
        if current_time >= outlier_ranges(w,1) && current_time <= outlier_ranges(w,2)
            is_outlier = true;
            break;
        end
    end
    
    if is_outlier
        sig_r = sig_r * 9;  
        sig_rr = sig_rr * 9;
    end
    
    R_true_hist(:, k) = [sig_r^2; sig_rr^2];
    noise_radar = [randn * sig_r; randn * sig_rr];
    
    Radar_Meas(:, k) = [dist; range_rate] + noise_radar;
end

% 6. 打包输出
SimData.Time = (0:TotalSteps-1)*dt;
SimData.dt = dt;
SimData.Pos_Radar = Pos_Radar;
SimData.X_true = true_state;
SimData.Bias_true = Bias_true;
SimData.GPS_Meas = GPS_Meas;
SimData.Radar_Meas = Radar_Meas;
SimData.R_true_hist = R_true_hist; 
SimData.Q_bias_std = sigma_bias_change;
SimData.outlier_ranges = outlier_ranges;

SimData.Q_cv = Q_cv; % 把目标的真实过程噪声传出来
SimData.X0_true = curr_state; % 传出目标的初始真实状态 (用于滤波器初始化)
end