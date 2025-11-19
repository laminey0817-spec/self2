% Step 1: 生成 U-Net 信道去噪训练数据
% clear; clc;
% setupPath;
% rng('default');
% 
% % 参数设置 (使用低SNR来训练去噪能力)
% optIn.SNRList = [10, 15, 20, 25, 30]; % 混合 SNR
% SNRList = optIn.SNRList;
% numSNR = length(optIn.SNRList);
% optIn.cycle = 50; % 生成足够多的数据 (例如 50 个 case)
% optIn.M = 4; optIn.N = 4; optIn.k = 1; optIn.RB = 4;
% optIn.Priority_queue = 0; optIn.Priority_proportion = 4/5;
% optIn.maxTrials = 3; % 加快速度
% optIn.inIt = 50;    % 加快速度
% optIn.outIt = 2;
% optIn.L = 480/optIn.RB; optIn.J = 48/optIn.RB;
% optIn.threshold_var_revise = 0.1; optIn.length_t = 10;
% optIn.alphabet = [1,1i,-1,-1i]; optIn.spar = 1-1e-12; optIn.Avar = 1;
% 
% % 构建 ML 矩阵 (复制自 simulate.m)
% global alphabet; alphabet = optIn.alphabet;
% % ... (ML矩阵构建代码简略，实际需完整复制) ...
% temp0=[1 2 3 4];temp1=ones(1,64);temp2=ones(1,16);t21=[temp2 2*temp2 3*temp2 4*temp2];
% temp3=ones(1,4);t31=[temp3 2*temp3 3*temp3 4*temp3];t32=[t31 t31 t31 t31];
% t41=[temp0 temp0 temp0 temp0];t42=[t41 t41 t41 t41];
% X_ML_matrix(1,:)=[temp1 2*temp1 3*temp1 4*temp1];X_ML_matrix(2,:)=[t21 t21 t21 t21];
% X_ML_matrix(3,:)=[t32 t32 t32 t32];X_ML_matrix(4,:)=[t42 t42 t42 t42];
% for i=1:4; X_ML_matrix(X_ML_matrix==i)=alphabet(i); end
% optIn.X_ML_matrix = X_ML_matrix;
% 
% % 加载信道
% load('Channel_TDLA30_R.mat','H','R_1','R_2','R_3','R_4','average_variance_H_A');
% optIn.average_variance_H = average_variance_H_A;
% optIn.var_slope = optIn.average_variance_H;
% optIn.R_1 = R_1; optIn.R_2 = R_2; optIn.R_3 = R_3; optIn.R_4 = R_4;
% 
% Global_H_Noisy = [];
% Global_H_Clean = [];
% 
% fprintf('开始生成 U-Net 训练数据...\n');
% 
% idx = 1; jdx = 1;
% while idx <= optIn.cycle
%     disp(['Generating Case: ', num2str(idx)]);
%     while (jdx <= numSNR)
%         optIn.SNR = SNRList(jdx);
%         [Signal,optIn] = generateSignal(optIn, H,idx);
% 
%         % BiG-AMP
%         for ii = 1:optIn.RB
%             optIn.W = (randn(optIn.M, optIn.L)+randn(optIn.M, optIn.L)*1i) ./ sqrt(2) * sqrt(10^(-SNRList(jdx)/10));
%             optIn.Y = Signal.Y_nonoise(:,:,ii) + optIn.W;
%             optIn.Y_allRB(:,:,ii) = optIn.Y;
%             optIn.Wvar = 10^(-SNRList(jdx)/10);
%             [results1,optIn] = parallel_BiGAMP(optIn);
%             result_all1.xhat(:,:,ii) = results1.xhat_final;
%             result_all1.xvar(:,:,ii) = results1.xvar_final;
%             result_all1.Ahat(:,:,ii) = results1.Ahat_final; 
%             result_all1.sum_EM_cycle(ii) = results1.sum_EM_cycle;
%             result_all1.sum_BiG_cycle(ii) = results1.sum_BiG_cycle;            
%         end
% 
%         [results2,result_all1] = revise(optIn,result_all1);
% 
%         % === 收集数据 ===
%         % Input: 粗糙信道 (4 x 4 x 48)
%         H_input = results2.H_LMMSE_raw; 
% 
%         % Target: 真实信道 (4 x 4 x 48 x 10) -> 时间平均 -> (4 x 4 x 48)
%         % 我们训练网络去预测“静态”信道部分，忽略快速时变
%         H_target_full = Signal.H_real; 
%         H_target = mean(H_target_full, 4); 
% 
%         % 将 Tx-Rx 链路拆开，增加样本量
%         % 4x4=16 条链路，每条 48 个子载波
%         % Data Shape: [48, 1, 2, 16] (H, W, C, Batch)
%         for tx = 1:4
%             for rx = 1:4
%                 h_in_vec = squeeze(H_input(rx, tx, :)); % 48x1
%                 h_out_vec = squeeze(H_target(rx, tx, :)); % 48x1
% 
%                 % 特征: 实部, 虚部
%                 feat_in = cat(3, real(h_in_vec), imag(h_in_vec)); % 48x1x2
%                 feat_out = cat(3, real(h_out_vec), imag(h_out_vec)); % 48x1x2
% 
%                 Global_H_Noisy = cat(4, Global_H_Noisy, feat_in);
%                 Global_H_Clean = cat(4, Global_H_Clean, feat_out);
%             end
%         end
% 
%         jdx = jdx + 1;
%     end
%     jdx = 1; idx = idx + 1;
% end
% 
% % 保存
% XTrain = Global_H_Noisy;
% YTrain = Global_H_Clean;
% % 简单划分验证集
% numVal = floor(size(XTrain,4) * 0.2);
% XVal = XTrain(:,:,:,1:numVal);
% YVal = YTrain(:,:,:,1:numVal);
% XTrain = XTrain(:,:,:,numVal+1:end);
% YTrain = YTrain(:,:,:,numVal+1:end);
% 
% save('unet_channel_dataset.mat', 'XTrain', 'YTrain', 'XVal', 'YVal');
% fprintf('数据生成完成，样本数: %d\n', size(XTrain, 4));


% Step 1: 生成 U-Net 信道去噪训练数据 (改进版：覆盖高SNR)
clear; clc;
setupPath;
rng('default');

% 参数设置
% 【修改】增加 20, 25, 30dB，让网络学会高信噪比下“不要乱动”
optIn.SNRList = [10, 15, 20, 25, 30]; 
SNRList = optIn.SNRList;
numSNR = length(optIn.SNRList);

% 增加 case 数量以获得更多样本
optIn.cycle = 50; 
optIn.M = 4; optIn.N = 4; optIn.k = 1; optIn.RB = 4;
optIn.Priority_queue = 0; optIn.Priority_proportion = 4/5;
optIn.maxTrials = 3; 
optIn.inIt = 50;    
optIn.outIt = 2;
optIn.L = 480/optIn.RB; optIn.J = 48/optIn.RB;
optIn.threshold_var_revise = 0.1; optIn.length_t = 10;
optIn.alphabet = [1,1i,-1,-1i]; optIn.spar = 1-1e-12; optIn.Avar = 1;

% 构建 ML 矩阵
global alphabet; alphabet = optIn.alphabet;
temp0=[1 2 3 4];temp1=ones(1,64);temp2=ones(1,16);t21=[temp2 2*temp2 3*temp2 4*temp2];
temp3=ones(1,4);t31=[temp3 2*temp3 3*temp3 4*temp3];t32=[t31 t31 t31 t31];
t41=[temp0 temp0 temp0 temp0];t42=[t41 t41 t41 t41];
X_ML_matrix(1,:)=[temp1 2*temp1 3*temp1 4*temp1];X_ML_matrix(2,:)=[t21 t21 t21 t21];
X_ML_matrix(3,:)=[t32 t32 t32 t32];X_ML_matrix(4,:)=[t42 t42 t42 t42];
for i=1:4; X_ML_matrix(X_ML_matrix==i)=alphabet(i); end
optIn.X_ML_matrix = X_ML_matrix;

% 加载信道
load('Channel_TDLA30_R.mat','H','R_1','R_2','R_3','R_4','average_variance_H_A');
optIn.average_variance_H = average_variance_H_A;
optIn.var_slope = optIn.average_variance_H;
optIn.R_1 = R_1; optIn.R_2 = R_2; optIn.R_3 = R_3; optIn.R_4 = R_4;

Global_H_Noisy = [];
Global_H_Clean = [];

fprintf('开始生成 U-Net 训练数据 (涵盖 10-30dB)...\n');

idx = 1; jdx = 1;
while idx <= optIn.cycle
    disp(['Generating Case: ', num2str(idx), ' / ', num2str(optIn.cycle)]);
    while (jdx <= numSNR)
        optIn.SNR = SNRList(jdx);
        [Signal,optIn] = generateSignal(optIn, H,idx);

        % BiG-AMP
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

        [results2,result_all1] = revise(optIn,result_all1);

        % === 收集数据 ===
        H_input = results2.H_LMMSE_raw; 
        H_target_full = Signal.H_real; 
        H_target = mean(H_target_full, 4); 

        for tx = 1:4
            for rx = 1:4
                h_in_vec = squeeze(H_input(rx, tx, :)); % 48x1
                h_out_vec = squeeze(H_target(rx, tx, :)); % 48x1

                % 特征: 实部, 虚部
                feat_in = cat(3, real(h_in_vec), imag(h_in_vec)); % 48x1x2
                feat_out = cat(3, real(h_out_vec), imag(h_out_vec)); 

                Global_H_Noisy = cat(4, Global_H_Noisy, feat_in);
                Global_H_Clean = cat(4, Global_H_Clean, feat_out);
            end
        end
        jdx = jdx + 1;
    end
    jdx = 1; idx = idx + 1;
end

% 保存
XTrain = Global_H_Noisy;
YTrain = Global_H_Clean;
% 简单划分验证集 (20%)
numVal = floor(size(XTrain,4) * 0.2);
XVal = XTrain(:,:,:,1:numVal);
YVal = YTrain(:,:,:,1:numVal);
XTrain = XTrain(:,:,:,numVal+1:end);
YTrain = YTrain(:,:,:,numVal+1:end);

save('unet_channel_dataset.mat', 'XTrain', 'YTrain', 'XVal', 'YVal');
fprintf('数据生成完成，训练样本数: %d, 验证样本数: %d\n', size(XTrain, 4), size(XVal, 4));
