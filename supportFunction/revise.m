function  [results2,result_all1] = revise(optIn,result_all1)
N = optIn.N;
M = optIn.M;
L = optIn.L;
J = optIn.J;
RB = optIn.RB; 
Y_allRB = optIn.Y_allRB; 
X_allRB = optIn.X_allRB1; 
threshold_var_revise = optIn.threshold_var_revise; 
sigma2 = optIn.Wvar; 
X_ML_matrix = optIn.X_ML_matrix; 

length_f = optIn.L/10; 
length_t = optIn.length_t; 
xhat = result_all1.xhat; 
xvar = result_all1.xvar; 

% 自相关矩阵
R_1 = optIn.R_1;
R_2 = optIn.R_2;
R_3 = optIn.R_3;
R_4 = optIn.R_4;

%% 找到排列的顺序
compareX_Xhat = zeros(N,N,RB);
for ii = 1:RB
    for i = 1:N
        for i1 = 1:N
            compareX_Xhat(i,i1,ii) =  sum(X_allRB(i,:,ii) ~= xhat(i1,:,ii));
        end
    end
end

for ij = 1:RB
    compareX_Xhat_temp = compareX_Xhat(:,:,ij);
    for i = 1:N
        [n,~] = min(compareX_Xhat_temp,[],2);
        [~,p1] = min(n);
        [~,p2] = min(compareX_Xhat_temp(p1,:));
        
        P(i,:) = [p1,p2];
        compareX_Xhat_temp(p1,:) = L;
        compareX_Xhat_temp(:,p2) = L;
    end
    match_list = sortrows(P,1);
    positionH_X(:,ij)  = match_list(:,2); 
end

%% 消除排列的模糊性
for ii = 1:RB
    for ni = 1:N
        xhat_new(ni,:,ii) = xhat(positionH_X(ni,ii),:,ii); 
        xvar_new(ni,:,ii) = xvar(positionH_X(ni,ii),:,ii); 
    end
end

%% 拼回网格
for ii = 1:RB
    xhat_grid(:,:,:,ii) =  reshape(xhat_new(:,:,ii),[N,length_f,length_t]);  
    xvar_grid(:,:,:,ii) =  reshape(xvar_new(:,:,ii),[N,length_f,length_t]); 
    Y_grid(:,:,:,ii) =  reshape(Y_allRB(:,:,ii),[N,length_f,length_t]); 
end

% 拼回48*10的4RB的大网格中
for ii = 1:RB
    Xhat_grid( :,(ii-1)*length_f+1:ii*length_f,1:10,:) = xhat_grid(:,:,:,ii); 
    Xvar_grid( :,(ii-1)*length_f+1:ii*length_f,1:10,:) = xvar_grid(:,:,:,ii); 
    Y_grid_big( :,(ii-1)*length_f+1:ii*length_f,1:10,:) = Y_grid(:,:,:,ii); 
end

%% 逐载波信道估计
H_sub_nowhite = zeros(M,N,48); 
H_sub_wihte = zeros(M,N,48);

for ii = 1:48 
    Y_sub = squeeze(Y_grid_big(:,ii,:)); 
    X_sub = squeeze(Xhat_grid(:,ii,:)); 
    Xvar_sub = squeeze(Xvar_grid(:,ii,:)); 
    
    %% 不白化，直接做信道估计
    [xvar_max,~] = max(Xvar_sub,[],1); 
    [a,tag]=sort(xvar_max,'ascend');
    position_temp = tag(1:length_t*4/5);  
    
    % LMMSE
    x1 = squeeze(X_sub(:,position_temp))';
    y1 = squeeze(Y_sub(:,position_temp))';
    Hhat_temp = x1' * pinv((x1 * x1') + sigma2 * eye(size(y1,1))) * y1;
    H_sub_nowhite(:,:,ii) = Hhat_temp'; % 4 4 48
    
    %% 白化后再估计
    for iu = 1:N  
        Xvar_u = Xvar_sub(iu,:); 
        [a,tag]=sort(Xvar_u,'ascend');
        positon = tag(1:length_t*4/5);
        X_sub_trust = X_sub(:,positon);
        Y_sub_trust = Y_sub(:,positon);
        X_u = X_sub_trust(iu,:).'; 
        X_i = X_sub_trust; X_i(iu,:) = []; 
        X_I1 = X_i(1,:).'; X_I2 = X_i(2,:).'; X_I3 = X_i(3,:).';
        R_xx = 0.25 * (X_I1 * X_I1'+ X_I2 * X_I2'+ X_I3 * X_I3')+ sigma2 * eye(size(X_I1,1));
        P = pinv(sqrtm(R_xx));
        X_eq1 = (P * X_u).'; 
        Y_eq1 = (P * Y_sub_trust.').'; 
        W = 1/(sum(abs(X_eq1).^2,'all')/length(X_eq1));
        x1 = X_eq1'; y1 = Y_eq1';
        Hhat_temp = x1' * pinv((x1 * x1') + W * eye(size(y1,1))) * y1;
        H_sub_wihte(:,iu,ii) = Hhat_temp'; 
    end
end

%% 维纳滤波-无白化 (修正维度错误)
H_user1_Wiener = zeros(48, N); H_user2_Wiener = zeros(48, N);
H_user3_Wiener = zeros(48, N); H_user4_Wiener = zeros(48, N);

for iu = 1:N
    % user1
    H_user1 = squeeze(H_sub_nowhite(:,1,:)).';  
    W_p(:,:,iu) = squeeze(R_1(:,:,1))*pinv(squeeze(R_1(:,:,1))+(sigma2)*eye(size(R_1,1)));
    H_user1_Wiener(:,iu) = W_p(:,:,iu) * H_user1(:,iu);
    % user2
    H_user2 = squeeze(H_sub_nowhite(:,2,:)).';
    W_p(:,:,iu) = squeeze(R_2(:,:,2))*pinv(squeeze(R_2(:,:,2))+(sigma2)*eye(size(R_2,1)));
    H_user2_Wiener(:,iu) = W_p(:,:,iu) * H_user2(:,iu);
    % user3
    H_user3 = squeeze(H_sub_nowhite(:,3,:)).';
    W_p(:,:,iu) = squeeze(R_3(:,:,3))*pinv(squeeze(R_3(:,:,3))+(sigma2)*eye(size(R_3,1)));
    H_user3_Wiener(:,iu) = W_p(:,:,iu) * H_user3(:,iu);
    % user4
    H_user4 = squeeze(H_sub_nowhite(:,4,:)).';
    W_p(:,:,iu) = squeeze(R_4(:,:,4))*pinv(squeeze(R_4(:,:,4))+(sigma2)*eye(size(R_4,1)));
    H_user4_Wiener(:,iu) = W_p(:,:,iu) * H_user4(:,iu);
end

% 将逐天线的拼回
H_wiener_nowhite = zeros(M,N,48); 
H_wiener_nowhite(:,1,:) = H_user1_Wiener.';
H_wiener_nowhite(:,2,:) = H_user2_Wiener.';
H_wiener_nowhite(:,3,:) = H_user3_Wiener.';
H_wiener_nowhite(:,4,:) = H_user4_Wiener.';

%% 维纳滤波-有白化
H_user1_Wiener = zeros(48, N); H_user2_Wiener = zeros(48, N);
H_user3_Wiener = zeros(48, N); H_user4_Wiener = zeros(48, N);
for iu = 1:N
    H_user1 = squeeze(H_sub_wihte(:,1,:)).';  
    W_p(:,:,iu) = squeeze(R_1(:,:,1))*pinv(squeeze(R_1(:,:,1))+(sigma2)*eye(size(R_1,1)));
    H_user1_Wiener(:,iu) = W_p(:,:,iu) * H_user1(:,iu);
    
    H_user2 = squeeze(H_sub_wihte(:,2,:)).';
    W_p(:,:,iu) = squeeze(R_2(:,:,2))*pinv(squeeze(R_2(:,:,2))+(sigma2)*eye(size(R_2,1)));
    H_user2_Wiener(:,iu) = W_p(:,:,iu) * H_user2(:,iu);
    
    H_user3 = squeeze(H_sub_wihte(:,3,:)).';
    W_p(:,:,iu) = squeeze(R_3(:,:,3))*pinv(squeeze(R_3(:,:,3))+(sigma2)*eye(size(R_3,1)));
    H_user3_Wiener(:,iu) = W_p(:,:,iu) * H_user3(:,iu);
    
    H_user4 = squeeze(H_sub_wihte(:,4,:)).';
    W_p(:,:,iu) = squeeze(R_4(:,:,4))*pinv(squeeze(R_4(:,:,4))+(sigma2)*eye(size(R_4,1)));
    H_user4_Wiener(:,iu) = W_p(:,:,iu) * H_user4(:,iu);
end
H_wiener_white = zeros(M,N,48);
H_wiener_white(:,1,:) = H_user1_Wiener.';
H_wiener_white(:,2,:) = H_user2_Wiener.';
H_wiener_white(:,3,:) = H_user3_Wiener.';
H_wiener_white(:,4,:) = H_user4_Wiener.';

%% 逐载波的信号检测（修正）
X_wiener_zhuzaibo = Xhat_grid; 
for ii = 1:48
    H_sub = H_wiener_nowhite(:,:,ii); 
    Y_sub = squeeze(Y_grid_big(:,ii,:)); 
    Xvar_sub = squeeze(Xvar_grid(:,ii,:)); 
    [xvar_max,~] = max(Xvar_sub,[],1);
    positon_X_revise = find(xvar_max > threshold_var_revise); 
    if size(positon_X_revise,1) ~= 0 
        for ij = 1:length(positon_X_revise)
            Y_temp = Y_sub(:,positon_X_revise(ij)); 
            H_temp = H_sub; 
            Y_HX = zeros(1,256);
            for im = 1:256
                Y_HX(im) = norm((Y_temp - H_temp * X_ML_matrix(:,im)),'fro');
            end
            [~,minY_HXlocation] = min(Y_HX);
            X_wiener_zhuzaibo(:,ii,positon_X_revise(ij)) = X_ML_matrix(:,minY_HXlocation);
        end
    end
end

% 均值斜率模型
X_wiener = junzhixielv(xhat_new,xvar_new,Y_grid_big,Xhat_grid,Xvar_grid,optIn);
results2.X_junzhixielv_wiener = X_wiener; 

% 结果打包
results2.X_wiener_zhuzaibo = X_wiener_zhuzaibo;
results2.H_wiener_white = H_wiener_white; 
results2.H_wiener_nowhite = H_wiener_nowhite; 
results2.positionH_X = positionH_X; 

% === 【关键修改】导出粗糙信道用于 U-Net 输入 ===
results2.H_LMMSE_raw = H_sub_nowhite; % 4x4x48

result_all1.Xhat_grid_eliminat = Xhat_grid;
result_all1.Xvar_grid_eliminat = Xvar_grid;
end