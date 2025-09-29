%%  清空环境变量
warning off             % 关闭报警信息
close all               % 关闭开启的图窗
clear                   % 清空变量
clc                     % 清空命令行

%%  导入数据
res =xlsread('G:\2\data\SG_daoshu_yijie.xlsx','193转置','A2:BUC194');

%%  划分训练集和测试集
rng(3111)%随机种子
temp = randperm(193);

P_train = res(temp(1: 116), 1: 1900)';
T_train = res(temp(1: 116), 1901)';
M = size(P_train, 2);

P_test = res(temp(117: end), 1: 1900)';
T_test = res(temp(117: end), 1901)';
N = size(P_test, 2);

%%  数据归一化
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test = mapminmax('apply', P_test, ps_input);

[t_train, ps_output] = mapminmax(T_train, 0, 1);
t_test = mapminmax('apply', T_test, ps_output);

%%  转置以适应模型
p_train = p_train'; p_test = p_test';
t_train = t_train'; t_test = t_test';

%% 创建模型
c = 2.8;    % 惩罚因子 (BoxConstraint)
g = 24;    % 径向基函数参数 (KernelScale)

% 使用fitrsvm进行回归训练
model = fitrsvm(p_train, t_train, 'KernelFunction', 'rbf', 'BoxConstraint', c,'KernelScale',g);

%% 仿真预测
t_sim1 = predict(model, p_train);   % 训练集预测
t_sim2 = predict(model, p_test);    % 测试集预测

%%  数据反归一化
T_sim1 = mapminmax('reverse', t_sim1, ps_output);
T_sim2 = mapminmax('reverse', t_sim2, ps_output);

%%  均方根误差
error1 = sqrt(sum((T_sim1' - T_train).^2) ./ M);
error2 = sqrt(sum((T_sim2' - T_test ).^2) ./ N);

disp(['训练集数据的RMSE为：', num2str(error1)])
disp(['测试集数据的RMSE为：', num2str(error2)])

%%  绘图
%figure
%plot(1: M, T_train, 'r-*', 1: M, T_sim1, 'b-o', 'LineWidth', 1)
%legend('真实值', '预测值')
%xlabel('预测样本')
%ylabel('预测结果')
%string = {'训练集预测结果对比'; ['RMSE=' num2str(error1)]};
%title(string)
%xlim([1, M])
%grid

%figure
%plot(1: N, T_test, 'r-*', 1: N, T_sim2, 'b-o', 'LineWidth', 1)
%legend('真实值', '预测值')
%xlabel('预测样本')
%ylabel('预测结果')
%string = {'测试集预测结果对比'; ['RMSE=' num2str(error2)]};
%title(string)
%xlim([1, N])
%grid

%%  相关指标计算
% R2
R1 = 1 - norm(T_train - T_sim1')^2 / norm(T_train - mean(T_train))^2;
R2 = 1 - norm(T_test  - T_sim2')^2 / norm(T_test  - mean(T_test ))^2;

disp(['训练集数据的R2为：', num2str(R1)])
disp(['测试集数据的R2为：', num2str(R2)])

%RPD
rpd1 = [std(T_train)]/error1;
rpd2 = [std(T_test)]/error2;
disp(['训练集数据的RPD为：', num2str(rpd1)])
disp(['测试集数据的RPD为：', num2str(rpd2)])

% MAE
%mae1 = sum(abs(T_sim1' - T_train)) ./ M ;
%mae2 = sum(abs(T_sim2' - T_test )) ./ N ;

%disp(['训练集数据的MAE为：', num2str(mae1)])
%disp(['测试集数据的MAE为：', num2str(mae2)])

% MBE
%mbe1 = sum(T_sim1' - T_train) ./ M ;
%mbe2 = sum(T_sim2' - T_test ) ./ N ;

%disp(['训练集数据的MBE为：', num2str(mbe1)])
%disp(['测试集数据的MBE为：', num2str(mbe2)])

%%  绘制散点图
%sz = 25;
%c = 'b';

%figure
%scatter(T_train, T_sim1, sz, c)
%hold on
%plot(xlim, ylim, '--k')
%xlabel('训练集真实值');
%ylabel('训练集预测值');
%xlim([min(T_train) max(T_train)])
%ylim([min(T_sim1) max(T_sim1)])
%title('训练集预测值 vs. 训练集真实值')

%figure
%scatter(T_test, T_sim2, sz, c)
%hold on
%plot(xlim, ylim, '--k')
%xlabel('测试集真实值');
%ylabel('测试集预测值');
%xlim([min(T_test) max(T_test)])
%ylim([min(T_sim2) max(T_sim2)])
%title('测试集预测值 vs. 测试集真实值')

%%输出训练集和测试集的实测值及预测值
T_train2 = T_train(:); % 转换为列向量
T_test2 = T_test(:);   % 转换为列向量

max_len = max([length(T_train2), length(T_test2), length(T_sim1), length(T_sim2)]);
T_train2 = [T_train2; NaN(max_len - length(T_train2), 1)];
T_test2 = [T_test2; NaN(max_len - length(T_test2), 1)];
T_sim1 = [T_sim1; NaN(max_len - length(T_sim1), 1)];
T_sim2 = [T_sim2; NaN(max_len - length(T_sim2), 1)];

data_to_write = [T_train2, T_sim1, T_test2, T_sim2];
file_path = 'G:\2\data\SVM_result\CWT_SVM_result\R_SVM.xlsx';

writematrix(data_to_write, file_path, 'Sheet', 'sheet3', 'Range', 'A1'); 
