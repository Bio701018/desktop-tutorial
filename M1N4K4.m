clc
close all % 关闭所有的Figure窗口 
clearvars

Sample_num = 1e4; 
%% 模型参数
M = 1;       % AP个数
N = 16;       % AP天线数
K = 4;       % 用户数

%% 三段式大尺度衰落参数设置
D = 1000;      % 正方形区域边长（单位：米）
d0 = 125;      
d1 = 250;      
f = 1900;       
h_AP = 20;       
h_u = 1.65;      
L = 46.3 + 33.9*log10(f) - 13.82*log10(h_AP) - (1.1*log10(f)-0.7)*h_u + (1.56*log10(f)-0.8);
deta_sh_dB = 8;
deta_sh = deta_sh_dB; 
tau_c = 200; 
tau_p = K; 
yinta = ones(1,K); 

k_B = 1.381e-23;
T0 = 290;
noise_figure_dB = 9; 
noise_figure = 10^(noise_figure_dB/10);
B = 20e6; 
noise_power = B*k_B*T0*noise_figure;
% --- 统一使用 dBm 作为基准 ---
Pu_cf_dBm = 20; % 用户的物理发送功率为 20 dBm (即 100 mW)
Pp_cf_dBm = 20; % 导频的物理发送功率也为 20 dBm

% 将 dBm 转换为十进制的瓦特 (W)
% 公式：W = 10^((dBm - 30) / 10)
Pu_cf = 10^( (Pu_cf_dBm - 30) / 10 ); 
Pp_cf = 10^( (Pp_cf_dBm - 30) / 10 );

% 统一除以底噪进行归一化
pu_cf = Pu_cf / noise_power;
pp_cf = Pp_cf / noise_power;

% 存储中心化MMSE速率
rate_k_centralized_mmse = zeros(Sample_num,K);

%% 获取大尺度Beta与位置信息
[Beta, AP_Site, User_Site] = Beta_Caculate_AP_Change(Sample_num,M,K,D,L,d0,d1,deta_sh);  

%% 1. 绘制前 5 个样本的用户与 AP 分布图
figure('Name', 'AP与用户随机分布图 (前5个样本)', 'Position', [100, 100, 1200, 300]);
for i = 1:5
    subplot(1, 5, i);
    hold on; grid on; box on;
    
    % --- 1. 解析 AP 坐标 ---
    if isreal(AP_Site) && size(AP_Site, 2) >= 2
        ap_x = AP_Site(1,1); ap_y = AP_Site(1,2); 
    else
        ap_x = real(AP_Site(1)); ap_y = imag(AP_Site(1)); 
    end
    scatter(ap_x, ap_y, 120, 'r^', 'filled'); 
    
    % --- 2. 解析 User 坐标 ---
    if isreal(User_Site)
        if ndims(User_Site) == 3 && size(User_Site, 2) >= 2
            % 如果是用 [K, 2, Sample] 格式存储的 [X, Y]
            user_x = User_Site(:, 1, i);
            user_y = User_Site(:, 2, i);
        else
            % 如果全是实数且只有一维，说明 Y 坐标丢失了
            user_x = User_Site(:, i);
            user_y = zeros(K, 1);
            if i == 1
                disp('⚠️ 警告: User_Site 全是实数，说明你的外部函数可能漏乘了 1j，导致用户 Y 坐标丢失！');
            end
        end
    else
        % 标准复数格式: X + jY
        user_x = real(User_Site(:, i));
        user_y = imag(User_Site(:, i));
    end
    scatter(user_x, user_y, 40, 'bo', 'filled');
    
    title(['样本 ', num2str(i)]);
    xlabel('X (m)'); ylabel('Y (m)');
    
    % --- 3. 动态修正坐标轴范围 ---
    % 将坐标轴改为以原点为中心，适配 [-500, 500] 的范围
    axis([-D/2 D/2 -D/2 D/2]); 
    
    if i == 5
        legend('AP', 'Users', 'Location', 'best');
    end
end

%% 生成信道
H = sqrt(1/2)*(randn(M,K,N,Sample_num)+1j*randn(M,K,N,Sample_num));
H_realization = zeros(M,K,N,Sample_num);

for n = 1:Sample_num
    for m = 1:M
        for k = 1:K
            H_realization(m,k,:,n) = sqrt(Beta(m,k,n))*H(m,k,:,n);
        end
    end
end

H_perfect = H_realization;
[~, V_MMSE_centralized] = Beamforming_caculate_Change(H_perfect,Beta,Sample_num,M,K,N,tau_p,pp_cf);

H_mk = permute(H_realization,[1,3,2,4]);           % M×N×K×Sample_num

temp1 = zeros(M,N,Sample_num,K);  
for m = 1:M
    temp1(m,:,:,:) = V_MMSE_centralized((m-1)*N+1:m*N,:,:);
end
V_MMSE_Centralized = permute(temp1,[1,2,4,3]); % M×N×K×Sample_num

%% 2. 初始化用于保存数据集的变量并进行收发仿真
% 生成原始随机数据比特流，并映射为 QPSK 符号序列 s
bits_stream = randi([0 1], K, 2, Sample_num); 
s_symbols = (1 - 2*bits_stream(:,1,:)) + 1j*(1 - 2*bits_stream(:,2,:));
s_symbols = s_symbols / sqrt(2); % 功率归一化，维度: K × 1 × Sample_num

Y_received = zeros(M*N, Sample_num);
S_hat_save = zeros(K, Sample_num);
H_est_save = zeros(M*N, K, Sample_num);

for n = 1:Sample_num
    rate_k_centralized_mmse(n,:) = Rate_caculate_M(H_mk(:,:,:,n),V_MMSE_Centralized(:,:,:,n),K,M,yinta,pu_cf);
    
    % --- 信号接收与估计环节 ---
    % 提取当前样本的真实信道和估计信道矩阵 (MN × K)
    H_true_n = reshape(H_mk(:,:,:,n), M*N, K);
    H_est_n = H_true_n;
    H_est_save(:,:,n) = H_est_n;
    
    % 提取当前样本的用户发送符号 (K ×1) 
    s_n = s_symbols(:, 1, n);
    
    % 生成归一化高斯白噪声 n (MN × 1)
    noise = sqrt(1/2)*(randn(M*N,1) + 1j*randn(M*N,1));
    
    % AP 接收信号 y = sqrt(p_u) * H * s + n
    y_n = sqrt(pu_cf) * H_true_n * s_n + noise;
    Y_received(:,n) = y_n;
    
    % 使用 Centralized-MMSE 接收矩阵进行符号估计: s^ = V^H * y
    % 提取当前 V 矩阵并调整维度为 (MN × K)
    V_n = reshape(V_MMSE_Centralized(:,:,:,n), M*N, K);
    s_hat_n = V_n' * y_n;
    S_hat_save(:,n) = s_hat_n;
    
    % 打印进度
    if mod(n, 1000) == 0
        disp(['已处理样本 ' num2str(n) ' / ' num2str(Sample_num)]);
    end
end

%% 计算和速率并清洗
rate_centralized_mmse = ((tau_c-tau_p)/tau_c)*sum(rate_k_centralized_mmse,2);

%% 3. 保存构建好的数据集为 v7.3 格式
% 整理变量维度以方便机器学习读取
s = squeeze(s_symbols);  % 去除多余维度，变为 K × Sample_num
H_est = H_est_save;      % MN × K × Sample_num
Y = Y_received;          % MN × Sample_num
S_hat = S_hat_save;      % K × Sample_num

save('Dataset__M1N4K4.mat', 'H_est', 'Y', 's', 'S_hat', 'bits_stream', '-v7.3');

disp('数据集 Dataset_M1N4K4.mat 已成功保存为 v7.3 格式！');
