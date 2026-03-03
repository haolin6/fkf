% =========================================================================
% main_mc_comparison.m
% 蒙特卡洛实验主程序：GPS/Radar 组合导航算法对比
% 功能：调用 get_simulation_data 生成数据，调用各 alg_xxx 函数进行滤波
% =========================================================================
clc; clear; close all;

%% 1. 实验配置
MC_Runs = 20;           % 蒙特卡洛实验次数
Run_Algorithms = { ...  % 选择要运行的算法开关
    'Raw', ...          % 未滤波 (原始GPS)
    'SRCKF', ...        % 标准 SRCKF
    'VB-SRCKF', ...     % 变分贝叶斯 SRCKF
    'FKF', ...          % 鲁棒 FKF
    'VB-FKF', ...       % 变分贝叶斯 FKF (方法3)
    'Adaptive-FKF'      % 【新增】基于残差的自适应 FKF
};
Run_Algorithms = { ...  % 选择要运行的算法开关
    'Raw', ...          % 未滤波 (原始GPS)
    'SRCKF', ...        % 标准 SRCKF
    'FKF', ...          % 鲁棒 FKF
    'Adaptive-FKF'      % 【新增】基于残差的自适应 FKF
};

fprintf('=== 开始蒙特卡洛实验 (总次数: %d) ===\n', MC_Runs);

%% 2. 初始化存储空间
% 预运行一次以获取数据维度
demo_data = get_simulation_data(1); 
Time = demo_data.Time;
TotalSteps = length(Time);
outlier_ranges = demo_data.outlier_ranges;

% 初始化平方误差和 (Sum of Squared Errors) 容器
SSE.Raw          = zeros(1, TotalSteps);
SSE.SRCKF        = zeros(1, TotalSteps);
SSE.VB_SRCKF     = zeros(1, TotalSteps);
SSE.FKF          = zeros(1, TotalSteps);
SSE.VB_FKF       = zeros(1, TotalSteps); 
SSE.Adaptive_FKF = zeros(1, TotalSteps); % 【新增】初始化 Adaptive-FKF 容器

% 进度条
h_wait = waitbar(0, 'Monte Carlo Simulation Running...');

%% 3. 蒙特卡洛循环
t_start = tic;

for m = 1:MC_Runs
    % 3.1 生成本次实验数据 (传入 m 作为随机种子)
    SimData = get_simulation_data(m);
    
    % 提取本次实验的真值和量测
    X_True = SimData.X_true;       % 真实位置/速度
    GPS_Meas = SimData.GPS_Meas;   % GPS 量测
    
    % ---------------------------------------------------------
    % 3.2 算法 0: Raw GPS (基准)
    % ---------------------------------------------------------
    pos_err_vec = GPS_Meas(1:3, :) - X_True(1:3, :);
    SSE.Raw = SSE.Raw + sum(pos_err_vec.^2, 1); 
    
    % ---------------------------------------------------------
    % 3.3 算法 1: Standard SRCKF
    % ---------------------------------------------------------
    if any(strcmp(Run_Algorithms, 'SRCKF'))
        [X_est_bias, ~] = alg_srckf(SimData);
        Pos_Corrected = GPS_Meas(1:3, :) - X_est_bias(1:3, :);
        err_vec = Pos_Corrected - X_True(1:3, :);
        SSE.SRCKF = SSE.SRCKF + sum(err_vec.^2, 1);
    end
    
    % ---------------------------------------------------------
    % 3.4 算法 2: VB-SRCKF
    % ---------------------------------------------------------
    if any(strcmp(Run_Algorithms, 'VB-SRCKF'))
        [X_est_bias, ~] = alg_vbsrckf(SimData);
        Pos_Corrected = GPS_Meas(1:3, :) - X_est_bias(1:3, :);
        err_vec = Pos_Corrected - X_True(1:3, :);
        SSE.VB_SRCKF = SSE.VB_SRCKF + sum(err_vec.^2, 1);
    end
    
    % ---------------------------------------------------------
    % 3.5 算法 3: FKF (Robust)
    % ---------------------------------------------------------
    if any(strcmp(Run_Algorithms, 'FKF'))
        [X_est_bias, ~] = alg_fkf(SimData);
        Pos_Corrected = GPS_Meas(1:3, :) - X_est_bias(1:3, :);
        err_vec = Pos_Corrected - X_True(1:3, :);
        SSE.FKF = SSE.FKF + sum(err_vec.^2, 1);
    end
    
    % ---------------------------------------------------------
    % 3.6 算法 4: VB-FKF (New Adaptive Robust)
    % ---------------------------------------------------------
    if any(strcmp(Run_Algorithms, 'VB-FKF'))
        [X_est_vbfkf, ~] = alg_vbfkf(SimData);
        Pos_Corrected = GPS_Meas(1:3, :) - X_est_vbfkf(1:3, :);
        err_vec = Pos_Corrected - X_True(1:3, :);
        SSE.VB_FKF = SSE.VB_FKF + sum(err_vec.^2, 1); 
    end

    % ---------------------------------------------------------
    % 3.7 算法 5: Adaptive-FKF (Residual-Based)
    % ---------------------------------------------------------
    if any(strcmp(Run_Algorithms, 'Adaptive-FKF'))
        [X_est_adafkf, ~] = alg_adaptive_fkf(SimData); % 【调用我们上一轮写好的函数】
        Pos_Corrected = GPS_Meas(1:3, :) - X_est_adafkf(1:3, :);
        err_vec = Pos_Corrected - X_True(1:3, :);
        SSE.Adaptive_FKF = SSE.Adaptive_FKF + sum(err_vec.^2, 1); 
    end

    % 更新进度条
    if mod(m, 5) == 0
        waitbar(m/MC_Runs, h_wait, sprintf('Running Run %d / %d (Time: %.1fs)', m, MC_Runs, toc(t_start)));
    end
end
close(h_wait);
fprintf('实验完成，耗时: %.2f 秒\n', toc(t_start));

%% 4. 计算 RMSE 并绘图
% RMSE = sqrt( Sum(Error^2) / M )
RMSE.Raw          = sqrt(SSE.Raw / MC_Runs);
RMSE.SRCKF        = sqrt(SSE.SRCKF / MC_Runs);
RMSE.VB_SRCKF     = sqrt(SSE.VB_SRCKF / MC_Runs);
RMSE.FKF          = sqrt(SSE.FKF / MC_Runs);
RMSE.VB_FKF       = sqrt(SSE.VB_FKF / MC_Runs); 
RMSE.Adaptive_FKF = sqrt(SSE.Adaptive_FKF / MC_Runs); % 【新增】

% --- 绘图 ---
figure('Position', [100, 100, 1000, 600], 'Color', 'w');
hold on; box on;

% 1. 绘制粗差区间背景
y_max_plot = max(RMSE.Raw) * 1.1; 
for w = 1:size(outlier_ranges, 1)
    x_patch = [outlier_ranges(w,1), outlier_ranges(w,2), outlier_ranges(w,2), outlier_ranges(w,1)];
    y_patch = [0, 0, y_max_plot, y_max_plot];
    fill(x_patch, y_patch, [1 0.9 0.9], 'EdgeColor', 'none', 'FaceAlpha', 0.5, 'HandleVisibility', 'off');
    text(mean(outlier_ranges(w,:)), y_max_plot*0.9, 'Outlier', 'Color', 'r', 'HorizontalAlignment', 'center');
end

% 2. 绘制 RMSE 曲线
plot(Time, RMSE.Raw,          'Color', [0.7 0.7 0.7], 'LineWidth', 1.5, 'DisplayName', 'Raw GPS');
plot(Time, RMSE.SRCKF,        'b-.', 'LineWidth', 1.5, 'DisplayName', 'Standard SRCKF');
plot(Time, RMSE.VB_SRCKF,     'g-',  'LineWidth', 2.0, 'DisplayName', 'VB-SRCKF');
plot(Time, RMSE.FKF,          'r-',  'LineWidth', 2.0, 'DisplayName', 'FKF (Robust)');
plot(Time, RMSE.VB_FKF,       'm-',  'LineWidth', 2.0, 'DisplayName', 'VB-FKF (Adaptive)'); 
% 【新增】给 Adaptive-FKF 选一个醒目的颜色，比如暗橙色或深青色
plot(Time, RMSE.Adaptive_FKF, 'Color', [0.8500 0.3250 0.0980], 'LineWidth', 2.5, 'DisplayName', 'Adaptive-FKF (Residual)'); 

xlabel('Time (s)'); ylabel('Position RMSE (m)');
title(sprintf('Monte Carlo Simulation Results (Runs=%d)', MC_Runs));
legend('Location', 'best');
grid on;
xlim([0 Time(end)]);
ylim([0 y_max_plot]);

%% 5. 终端打印统计结果
fprintf('\n=========================================\n');
fprintf('       Algorithm Performance (RMSE)      \n');
fprintf('=========================================\n');
fprintf('Raw GPS      : %.4f m\n', mean(RMSE.Raw));
fprintf('SRCKF        : %.4f m\n', mean(RMSE.SRCKF));
fprintf('VB-SRCKF     : %.4f m\n', mean(RMSE.VB_SRCKF));
fprintf('FKF          : %.4f m\n', mean(RMSE.FKF));
fprintf('VB-FKF       : %.4f m\n', mean(RMSE.VB_FKF)); 
fprintf('Adaptive-FKF : %.4f m\n', mean(RMSE.Adaptive_FKF)); % 【新增】
fprintf('-----------------------------------------\n'); 

% 打印粗差区间的局部 RMSE
fprintf('Outlier Zones RMSE:\n');
in_outlier_mask = false(1, TotalSteps);
for w = 1:size(outlier_ranges, 1)
     idx = Time >= outlier_ranges(w,1) & Time <= outlier_ranges(w,2);
     in_outlier_mask = in_outlier_mask | idx;
end

fprintf('Raw GPS      : %.4f m\n', mean(RMSE.Raw(in_outlier_mask)));
fprintf('SRCKF        : %.4f m\n', mean(RMSE.SRCKF(in_outlier_mask)));
fprintf('VB-SRCKF     : %.4f m\n', mean(RMSE.VB_SRCKF(in_outlier_mask)));
fprintf('FKF          : %.4f m\n', mean(RMSE.FKF(in_outlier_mask)));
fprintf('VB-FKF       : %.4f m\n', mean(RMSE.VB_FKF(in_outlier_mask))); 
fprintf('Adaptive-FKF : %.4f m\n', mean(RMSE.Adaptive_FKF(in_outlier_mask))); % 【新增】
fprintf('=========================================\n');