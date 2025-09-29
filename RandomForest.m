clc
clear
%% 导入数据
X=xlsread('E:\建模数据\P0.01\193_shuxue_r.xlsx','daoshu_yijie','A3:UO195');%光谱数据
y=xlsread('E:\建模数据\P0.01\193_shuxue_r.xlsx','R','XT2:XT194');%化学值
%% 数据集划分
rng(3111)%随机种子，保持随机森林训练参数一致
R= randperm(193);
n=R(:,1:116);m=R(:,117:193);
% 训练集样本
train_data = X(n,:);
train_labels = y(n,:);
% 测试集样本
test_data = X(m,:);
test_labels= y(m,:);
%% 三：数据归一化
% 训练集和测试集输入：自带的归一化函
[train_in,inputps] = mapminmax(train_data');
train_in = train_in';
test_in = mapminmax('apply',test_data',inputps);
test_in = test_in';
% 测试集和测试集输出
[train_out,outputps] = mapminmax(train_labels');
train_out = train_out';
test_out = mapminmax('apply',test_labels',outputps);
test_out = test_out';

%%  随机森林模型参数设置及训练模型
trees =12; % 决策树数目
leaf  =4; % 最小叶子数
OOBPrediction = 'on';  % 打开误差图
OOBPredictorImportance = 'on'; % 计算特征重要性
Method = 'regression';  % 选择回归或分类
net = TreeBagger(trees, train_in, train_out, 'OOBPredictorImportance', OOBPredictorImportance,...
      'Method', Method, 'OOBPrediction', OOBPrediction, 'minleaf', leaf);
importance = net.OOBPermutedPredictorDeltaError;  % 重要性

%%  仿真测试
Predict_1 = predict(net, train_in );
Predict_2= predict(net, test_in );

%% 五：反归一化
predict_1= mapminmax('reverse',Predict_1,outputps);%训练集预测
predict_2=mapminmax('reverse',Predict_2,outputps);%测试集预测

%% 六：结果计算
num1=length(train_labels);%  训练集个数
num2=length(test_labels);%  测试集个数
% 计算训练集的决定系数R2
train_R2 = (num1* sum(predict_1 .* train_labels ) - sum(predict_1) * sum(train_labels ))^2 / ((num1 * sum((predict_1).^2) - (sum(predict_1))^2) * (num1* sum((train_labels ).^2) - (sum(train_labels))^2)); 
% 计算测试集的决定系数R2
test_R2= (num2* sum(predict_2 .* test_labels) - sum(predict_2) * sum(test_labels))^2 / ((num2 * sum((predict_2).^2) - (sum(predict_2))^2) * (num2* sum((test_labels).^2) - (sum(test_labels))^2));
% 预测均方误差
RMSEC=sqrt(sum((train_labels  - predict_1).^2)/num1);%  训练集均方根误差  
RMSEP=sqrt(sum((test_labels- predict_2).^2)/num2);%  测试集均方根误差 
% RPD
STDEV2=std(predict_2);
RPD2=STDEV2/RMSEP;% 测试集RPD
STDEV1=std(predict_1);
RPD1=STDEV2/RMSEC;%训练集RPD


%% 七：输出结果
fprintf('训练集决定系数: %.4f\n', train_R2);
fprintf('训练集均方根误差: %.4f\n', RMSEC);
fprintf('预测集决定系数: %.4f\n', test_R2);
fprintf('预测集均方根误差: %.4f\n', RMSEP);
fprintf('RPD1: %.4f\n',RPD1);
fprintf('RPD2: %.4f\n',RPD2);

% 假设train_R2, RMSEC, test_R2, RMSEP, RPD1, RPD2已经计算好
% 将数据组织成一个列向量
%output_data1 = [train_R2; RMSEC; RPD1; test_R2; RMSEP; RPD2];

% 创建一个单元数组，用于存储输出标题
%headers = {'Train R2', 'RMSEC', 'RPD1', 'Test R2', 'RMSEP', 'RPD2'};

% 创建输出矩阵
%output_matrix = [headers; num2cell(output_data1')];

% 将数据写入Excel文件
filename = 'E:\建模数据\P0.01\193_daoshu_yijie_r_RF.xlsx';  % Excel文件名
%xlswrite(filename, output_matrix, 'Sheet1', 'J1'); % 从A1单元格开始写入数据


% 找出最长列的长度
max_len = max([length(train_labels), length(test_labels), length(predict_1), length(predict_2)]);

% 将数据填充到相同长度，短的部分用NaN填充
train_labels_padded = [train_labels; nan(max_len - length(train_labels), 1)];
predict_1_padded = [predict_1; nan(max_len - length(predict_1), 1)];
test_labels_padded = [test_labels; nan(max_len - length(test_labels), 1)];
predict_2_padded = [predict_2; nan(max_len - length(predict_2), 1)];

% 将这些列按照顺序合并为一个矩阵
output_data = [train_labels_padded, predict_1_padded, test_labels_padded, predict_2_padded];

% 将数据写入Excel文件
xlswrite(filename, output_data, 'Sheet2', 'A1'); % A1是写入的起始单元格



% %% 八：画图
% figure
% plot(train_labels, predict_1, 'o', 'Color','b', 'MarkerFaceColor','none', 'MarkerSize', 5, 'MarkerEdgeColor', 'b');hold on
% plot(test_labels, predict_2, '*', 'Color','r', 'MarkerFaceColor','r', 'MarkerSize', 5);hold on
% plot([5 ,25], [5,25], 'k-', 'LineWidth', 1)
% legend( '训练集 （Calibration Set）','测试集 （Validation Set）')
% legend('boxoff')
% xlabel('实测（actual）' )
% ylabel('预测（Predicted）')
% box off