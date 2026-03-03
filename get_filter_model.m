function Model = get_filter_model(SimData, scenario_type)
% =========================================================================
% get_filter_model: 统一生成滤波器的状态方程、观测方程和初始化参数
% 输入: 
%   SimData - 仿真数据
%   scenario_type - 'BiasEstimation' (偏差估计) 或 'TargetTracking' (目标跟踪)
% 输出:
%   Model - 包含 X0, P0, Q, f_func, h_func 的结构体
% =========================================================================

    dt = SimData.dt;

    if strcmp(scenario_type, 'BiasEstimation')
        % -----------------------------------------------------------------
        % 场景 A：GPS 偏差估计 (平滑的随机游走)
        % -----------------------------------------------------------------
        Model.X0 = zeros(6,1);
        Model.P0 = diag([100, 100, 100, 1, 1, 1]); 
        Model.Q = diag(SimData.Q_bias_std.^2);
        
        % 状态转移：随机游走
        F_rw = eye(6);
        Model.f_func = @(x) F_rw * x;
        
        % 观测方程：注意这里多了一个参数 k (当前时间步)，以便在内部提取当前的 GPS
        Model.h_func = @(x, k) h_meas_bias(x, SimData.GPS_Meas(1:3,k), SimData.GPS_Meas(4:6,k), SimData.Pos_Radar);
        
    elseif strcmp(scenario_type, 'TargetTracking')
        % -----------------------------------------------------------------
        % 场景 B：直接目标跟踪 (高动态，强非线性)
        % -----------------------------------------------------------------
        % 使用真实的初始状态加上一定的初始误差
        Model.X0 = SimData.X_true(:, 1) + [randn(3,1)*50; randn(3,1)*5]; 
        Model.P0 = diag([1e4, 1e4, 1e4, 100, 100, 100]); 
        
        % 假设你在 SimData 里存了 Q_cv (或者在这里重写)
        % 这里直接给定一个极大的 Q 来匹配剧烈机动
        q_proc = 500; % 对应你实验里的极强机动
        Model.Q = diag([0.5*dt^2, 0.5*dt^2, 0.5*dt^2, dt, dt, dt].^2) * q_proc;
        
        % 状态转移：匀速直线运动 (CV)
        F_cv = eye(6); 
        F_cv(1,4) = dt; F_cv(2,5) = dt; F_cv(3,6) = dt;
        Model.f_func = @(x) F_cv * x;
        
        % 观测方程：纯雷达单站测距
        Model.h_func = @(x, k) h_meas_target(x, SimData.Pos_Radar);
        
    else
        error('未知的场景类型！');
    end
end

%% --- 内部物理观测函数 ---

% 1. 偏差估计场景的观测模型
function z = h_meas_bias(bias_state, gps_pos, gps_vel, radar_pos)
    pos_est = gps_pos - bias_state(1:3);
    vel_est = gps_vel - bias_state(4:6);
    diff = pos_est - radar_pos;
    dist = norm(diff);
    if dist < 1e-3, dist = 1e-3; end
    range_rate = dot(diff, vel_est) / dist;
    z = [dist; range_rate];
end

% 2. 直接目标跟踪的观测模型
function z = h_meas_target(target_state, radar_pos)
    pos_est = target_state(1:3);
    vel_est = target_state(4:6);
    diff = pos_est - radar_pos;
    dist = norm(diff);
    if dist < 1e-3, dist = 1e-3; end
    range_rate = dot(diff, vel_est) / dist;
    z = [dist; range_rate];
end