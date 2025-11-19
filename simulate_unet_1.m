% simulate_unet.m (UPnP 仿真脚本 - 最终修复版)
clear; clc;
setupPath;
rng('default');

%% === 加载 U-Net 模型 ===
if exist('unet_channel_model.mat', 'file')
    load('unet_channel_model.mat', 'unet_net');
    fprintf('加载 U-Net 模型成功。\n');
else
    error('未找到模型，请先运行 step2_train_unet.m');
end

%% === 参数设置 ===
optIn.SNRList = 10:5:25; 
SNRList = optIn.SNRList;
numSNR = length(optIn.SNRList);

optIn.cycle = 20;           % 循环次数
optIn.M = 4;                % 天线数
optIn.N = 4;                % 用户数
optIn.k = 1;                % 信道类型
optIn.RB = 4;               % RB分块数
optIn.Priority_queue = 0; 
optIn.Priority_proportion = 4/5;
optIn.maxTrials = 5; 
optIn.inIt = 200; 
optIn.outIt = 2;
optIn.L = 480/optIn.RB; 
optIn.J = 48/optIn.RB;
optIn.threshold_var_revise = 0.1; 
optIn.length_t = 10;
optIn.alphabet = [1,1i,-1,-1i]; 
optIn.spar = 1-1e-12; 
optIn.Avar = 1;
global alphabet; alphabet = optIn.alphabet;

%% === 构建 ML 矩阵 ===
temp0=[1 2 3 4];temp1=ones(1,64);temp2=ones(1,16);t21=[temp2 2*temp2 3*temp2 4*temp2];
temp3=ones(1,4);t31=[temp3 2*temp3 3*temp3 4*temp3];t32=[t31 t31 t31 t31];
t41=[temp0 temp0 temp0 temp0];t42=[t41 t41 t41 t41];
X_ML_matrix(1,:)=[temp1 2*temp1 3*temp1 4*temp1];X_ML_matrix(2,:)=[t21 t21 t21 t21];
X_ML_matrix(3,:)=[t32 t32 t32 t32];X_ML_matrix(4,:)=[t42 t42 t42 t42];
for i=1:4; X_ML_matrix(X_ML_matrix==i)=alphabet(i); end
optIn.X_ML_matrix = X_ML_matrix;

%% === 信道加载 ===
if optIn.k == 1
    load('Channel_TDLA30_R.mat','H','R_1','R_2','R_3','R_4','average_variance_H_A'); 
    optIn.average_variance_H = average_variance_H_A; 
end
optIn.var_slope = optIn.average_variance_H; 
optIn.R_1 = R_1; optIn.R_2 = R_2; optIn.R_3 = R_3; optIn.R_4 = R_4;

%% === 变量初始化 ===
BER1 = zeros(optIn.cycle,numSNR);       % 原始 BiG-AMP
BER2 = zeros(optIn.cycle,numSNR);       % 传统修正
BER_unet = zeros(optIn.cycle,numSNR);   % U-Net 修正
idx=1; jdx=1;

%% === 主循环 ===
while idx <= optIn.cycle
    disp(['Case: ', num2str(idx)]);

    while (jdx <= numSNR)
        optIn.SNR = SNRList(jdx);
        [Signal,optIn] = generateSignal(optIn, H,idx);

        % --- 1. Run BiG-AMP ---
        for ii = 1:optIn.RB
            optIn.W = (randn(optIn.M, optIn.L)+randn(optIn.M, optIn.L)*1i) ./ sqrt(2) * sqrt(10^(-SNRList(jdx)/10));
            optIn.Y = Signal.Y_nonoise(:,:,ii) + optIn.W;
            optIn.Y_allRB(:,:,ii) = optIn.Y;
            optIn.Wvar = 10^(-SNRList(jdx)/10);

            [results1,optIn] = parallel_BiGAMP(optIn);

            result_all1.xhat(:,:,ii) = results1.xhat_final;
            result_all1.xvar(:,:,ii) = results1.xvar_final;
            result_all1.Ahat(:,:,ii) = results1.Ahat_final; 
            result_all1.sum_EM_cycle(ii) = results1.sum_EM_cycle;
            result_all1.sum_BiG_cycle(ii) = results1.sum_BiG_cycle;            
        end

        % --- 2. Run Traditional Revise ---
        [results2,result_all1] = revise(optIn,result_all1);
        errRes = checkErrorBiGAMP(optIn,result_all1,results2,Signal);


        % --- 3. Run UPnP (U-Net Channel Denoising) ---
        H_raw = results2.H_LMMSE_raw; % 获取粗糙信道输入 (4x4x48)
        H_unet = zeros(size(H_raw));

        % U-Net 预测 (去噪)
        for tx = 1:4
            for rx = 1:4
                % 提取单条链路数据 (48x1)
                h_vec = squeeze(H_raw(rx, tx, :));
                % 构建输入特征: 48x1x2
                feat_in = cat(3, real(h_vec), imag(h_vec)); 

                % 网络推理
                feat_out = predict(unet_net, feat_in);

                % 重组复数信道
                H_unet(rx, tx, :) = feat_out(:,:,1) + 1i * feat_out(:,:,2);
            end
        end

        % 基于 H_unet 的 ML 符号检测
        % 重组接收信号 Y (4x48x10)
        length_f = 12; length_t = 10;
        Y_grid_big = zeros(4, 48, 10);
        for ii = 1:optIn.RB
             Y_grid_temp = reshape(optIn.Y_allRB(:,:,ii), [4, length_f, length_t]);
             Y_grid_big(:, (ii-1)*length_f+1:ii*length_f, :) = Y_grid_temp;
        end

        error_bits = 0;
        % 逐载波检测
        for f = 1:48 % 频域
            H_f = squeeze(H_unet(:,:,f)); % 4x4
            for t = 1:10 % 时域
                y_vec = Y_grid_big(:, f, t); % 4x1

                % ML 检测 (遍历所有可能符号组合)
                min_dist = inf;
                x_best = zeros(4,1);

                for k = 1:256
                    x_cand = X_ML_matrix(:, k);
                    d = norm(y_vec - H_f * x_cand)^2;
                    if d < min_dist
                        min_dist = d;
                        x_best = x_cand;
                    end
                end

                % 计算误码
                % 使用正确的真实符号变量 optIn.X_allRB
                x_true = optIn.X_allRB(:, f, t); 

                [~, ber] = biterr(pskdemod(x_best,4,0,'gray'), pskdemod(x_true,4,0,'gray'));
                error_bits = error_bits + ber;
            end
        end

        % 记录当前 Case 的结果
        BER_unet(idx, jdx) = error_bits / (4 * 48 * 10 * 2);
        BER1(idx,jdx) = errRes.BER1;
        BER2(idx,jdx) = errRes.BER2;

        disp(['SNR:',num2str(SNRList(jdx)),' - Revise:',num2str(BER2(idx,jdx)), ' - UNet:',num2str(BER_unet(idx,jdx))]);

        jdx = jdx + 1;
    end
    jdx = 1; idx = idx + 1;
end

%% === 绘图与结果分析 (顺序已修复) ===

% 1. 先计算各方案的平均 BER
avg_BER1 = mean(BER1, 1);          % 原始 BiG-AMP
avg_BER2 = mean(BER2, 1);          % 传统逐载波修正
avg_BER_unet = mean(BER_unet, 1);  % UPnP (深度学习修正)

% 2. 再计算混合策略 (Hybrid Strategy)
BER_hybrid = zeros(size(SNRList));
for i = 1:length(SNRList)
    % 混合策略阈值：12dB
    % 低于等于12dB信道恶劣，用U-Net；高于12dB信道良好，用传统方法
    if SNRList(i) <= 12 
        BER_hybrid(i) = avg_BER_unet(i); 
    else
        BER_hybrid(i) = avg_BER2(i);     
    end
end

% 3. 绘图
figure;
% 原始 BiG-AMP
semilogy(SNRList, avg_BER1, 'b-o', 'LineWidth', 1.5, 'DisplayName', 'BiG-AMP Original');
hold on;

% 传统修正
semilogy(SNRList, avg_BER2, 'r-s', 'LineWidth', 2, 'DisplayName', 'Traditional Revise');

% UPnP 深度学习修正
semilogy(SNRList, avg_BER_unet, 'k-p', 'LineWidth', 2, 'DisplayName', 'UPnP (U-Net)');

% 混合策略 (Optimal)
semilogy(SNRList, BER_hybrid, 'g-.', 'LineWidth', 2, 'DisplayName', 'Hybrid (Optimal)');

grid on;
legend('Location', 'southwest'); 
xlabel('SNR (dB)');
ylabel('Average Bit Error Rate (BER)');
title('Performance Comparison: BiG-AMP vs. Revise vs. UPnP');
