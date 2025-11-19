% %% Step 2: 训练信道去噪 U-Net
% clear; clc;
% load('unet_channel_dataset.mat');
% 
% % 输入尺寸: 48 (子载波) x 1 (宽度) x 2 (实虚部)
% inputSize = [48, 1, 2];
% 
% % 定义网络 (U-Net 结构)
% layers = [
%     imageInputLayer(inputSize, 'Name', 'input', 'Normalization', 'zscore')
% 
%     % Encoder 1
%     convolution2dLayer([3,1], 32, 'Padding', 'same', 'Name', 'enc1_c1')
%     reluLayer('Name', 'enc1_r1')
%     convolution2dLayer([3,1], 32, 'Padding', 'same', 'Name', 'enc1_c2')
%     reluLayer('Name', 'enc1_r2')
%     maxPooling2dLayer([2,1], 'Stride', [2,1], 'Name', 'enc1_pool') % 24x1
% 
%     % Encoder 2
%     convolution2dLayer([3,1], 64, 'Padding', 'same', 'Name', 'enc2_c1')
%     reluLayer('Name', 'enc2_r1')
%     convolution2dLayer([3,1], 64, 'Padding', 'same', 'Name', 'enc2_c2')
%     reluLayer('Name', 'enc2_r2')
%     maxPooling2dLayer([2,1], 'Stride', [2,1], 'Name', 'enc2_pool') % 12x1
% 
%     % Middle
%     convolution2dLayer([3,1], 128, 'Padding', 'same', 'Name', 'mid_c1')
%     reluLayer('Name', 'mid_r1')
%     dropoutLayer(0.2, 'Name', 'drop')
% 
%     % Decoder 2
%     transposedConv2dLayer([2,1], 64, 'Stride', [2,1], 'Name', 'dec2_up')
%     concatenationLayer(3, 2, 'Name', 'dec2_cat') % concat with enc2_r2
%     convolution2dLayer([3,1], 64, 'Padding', 'same', 'Name', 'dec2_c1')
%     reluLayer('Name', 'dec2_r1')
% 
%     % Decoder 1
%     transposedConv2dLayer([2,1], 32, 'Stride', [2,1], 'Name', 'dec1_up')
%     concatenationLayer(3, 2, 'Name', 'dec1_cat') % concat with enc1_r2
%     convolution2dLayer([3,1], 32, 'Padding', 'same', 'Name', 'dec1_c1')
%     reluLayer('Name', 'dec1_r1')
% 
%     % Output
%     convolution2dLayer([1,1], 2, 'Name', 'final_conv') % 恢复2通道
%     regressionLayer('Name', 'output')
% ];
% 
% lgraph = layerGraph(layers);
% lgraph = connectLayers(lgraph, 'enc2_r2', 'dec2_cat/in2');
% lgraph = connectLayers(lgraph, 'enc1_r2', 'dec1_cat/in2');
% 
% % 训练选项
% options = trainingOptions('adam', ...
%     'MaxEpochs', 100, ...
%     'MiniBatchSize', 64, ...
%     'InitialLearnRate', 0.001, ...
%     'Shuffle', 'every-epoch', ...
%     'ValidationData', {XVal, YVal}, ...
%     'ValidationFrequency', 50, ...
%     'Verbose', false, ...
%     'Plots', 'training-progress');
% 
% fprintf('开始训练 U-Net...\n');
% unet_net = trainNetwork(XTrain, YTrain, lgraph, options);
% 
% save('unet_channel_model.mat', 'unet_net');
% fprintf('模型保存成功！\n');

%% Step 2: 训练信道去噪 U-Net
clear; clc;
if ~exist('unet_channel_dataset.mat', 'file')
    error('找不到数据集，请先运行 step1_generate_unet_data.m');
end
load('unet_channel_dataset.mat');

% 输入尺寸: 48 (子载波) x 1 (宽度) x 2 (实虚部)
inputSize = [48, 1, 2];

% 定义网络 (U-Net 结构)
layers = [
    imageInputLayer(inputSize, 'Name', 'input', 'Normalization', 'zscore')

    % Encoder 1
    convolution2dLayer([3,1], 32, 'Padding', 'same', 'Name', 'enc1_c1')
    reluLayer('Name', 'enc1_r1')
    convolution2dLayer([3,1], 32, 'Padding', 'same', 'Name', 'enc1_c2')
    reluLayer('Name', 'enc1_r2')
    maxPooling2dLayer([2,1], 'Stride', [2,1], 'Name', 'enc1_pool') % 24x1

    % Encoder 2
    convolution2dLayer([3,1], 64, 'Padding', 'same', 'Name', 'enc2_c1')
    reluLayer('Name', 'enc2_r1')
    convolution2dLayer([3,1], 64, 'Padding', 'same', 'Name', 'enc2_c2')
    reluLayer('Name', 'enc2_r2')
    maxPooling2dLayer([2,1], 'Stride', [2,1], 'Name', 'enc2_pool') % 12x1

    % Middle
    convolution2dLayer([3,1], 128, 'Padding', 'same', 'Name', 'mid_c1')
    reluLayer('Name', 'mid_r1')
    dropoutLayer(0.1, 'Name', 'drop')

    % Decoder 2
    transposedConv2dLayer([2,1], 64, 'Stride', [2,1], 'Name', 'dec2_up')
    concatenationLayer(3, 2, 'Name', 'dec2_cat') % concat with enc2_r2
    convolution2dLayer([3,1], 64, 'Padding', 'same', 'Name', 'dec2_c1')
    reluLayer('Name', 'dec2_r1')

    % Decoder 1
    transposedConv2dLayer([2,1], 32, 'Stride', [2,1], 'Name', 'dec1_up')
    concatenationLayer(3, 2, 'Name', 'dec1_cat') % concat with enc1_r2
    convolution2dLayer([3,1], 32, 'Padding', 'same', 'Name', 'dec1_c1')
    reluLayer('Name', 'dec1_r1')

    % Output
    convolution2dLayer([1,1], 2, 'Name', 'final_conv') % 恢复2通道
    regressionLayer('Name', 'output')
];

lgraph = layerGraph(layers);
lgraph = connectLayers(lgraph, 'enc2_r2', 'dec2_cat/in2');
lgraph = connectLayers(lgraph, 'enc1_r2', 'dec1_cat/in2');

% 训练选项
options = trainingOptions('adam', ...
    'MaxEpochs', 50, ...
    'MiniBatchSize', 128, ...
    'InitialLearnRate', 0.001, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {XVal, YVal}, ...
    'ValidationFrequency', 50, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

fprintf('开始训练 U-Net...\n');
unet_net = trainNetwork(XTrain, YTrain, lgraph, options);

save('unet_channel_model.mat', 'unet_net');
fprintf('模型保存成功！\n');