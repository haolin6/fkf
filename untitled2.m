% =========================================================================
% verify_trajectory.m
% 验证"协方差坍塌"与"圆心假说"：单次实验 X轴偏差估计轨迹透视
% =========================================================================
clc; clear; close all;

fprintf('正在运行单次仿真以提取内部轨迹...\n');

% 1. 生成单次数据 (确保你的 get_simulation_data.m 里 15s 处有跳变)
SimData = get_simulation_data(1); % 取种子 1 跑单次
Time = SimData.Time;

% 提取真实的 X 轴偏差
Bias_true_X = SimData.Bias_true(1, :); 

% 2. 运行滤波器 (获取全时段的状态估计矩阵)
[X_SRCKF, ~]   = alg_srckf(SimData);
[X_FKF, ~]     = alg_fkf(SimData);
[X_AdaFKF, ~]  = alg_adaptive_fkf(SimData);

% 3. 绘制单维度 (X轴) 的状态估计轨迹
figure('Color', 'w', 'Position', [100, 100, 900, 500]);
hold on; box on; grid on;

% 绘制真实偏差的阶跃轨迹 (真值)
plot(Time, Bias_true_X, 'k-', 'LineWidth', 2.5, 'DisplayName', 'True Bias (真实偏差 X轴)');

% 绘制各滤波器的估计轨迹
plot(Time, X_SRCKF(1, :), 'b-.', 'LineWidth', 2, 'DisplayName', 'SRCKF 估计轨迹');
plot(Time, X_FKF(1, :), 'r-', 'LineWidth', 2, 'DisplayName', 'FKF 估计轨迹');
plot(Time, X_AdaFKF(1, :), 'Color', [0.8500 0.3250 0.0980], 'LineWidth', 2.5, 'DisplayName', 'Adaptive-FKF 估计轨迹');

% 标注关键时间点
xline(10, 'm--', '10s (CPA / 几何非线性爆发点)', 'LabelVerticalAlignment', 'bottom', 'LineWidth', 1.5);
xline(15, 'g--', '15s (真实偏差跳变点)', 'LabelVerticalAlignment', 'bottom', 'LineWidth', 1.5);

xlabel('Time (s)');
ylabel('Bias X-axis (m)');
title('单次运行透视：X轴偏差估计的真实轨迹 (揭示协方差坍塌)');
legend('Location', 'best');
xlim([0, 30]); % 只看前 30s 的精彩博弈