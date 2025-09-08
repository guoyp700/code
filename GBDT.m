%% 清空环境变量
warning off;               % 关闭报警信息
close all;                 % 关闭所有图窗
clear;                     % 清空变量
clc;                       % 清空命令行

% 读取数据
data = readmatrix('E:\data\P0.001\daoshu_yijie_CWT_7\daoshu_yijie_CWT_7_data.xlsx', 'Sheet', 'Sheet1', 'Range', 'A2:VF194');
X = data(:, 1:end-1);      % 提取特征
y = data(:, end);          % 提取目标值

% 参数设置
num_trees = 80;          % 迭代次数（树的数量）
learning_rate = 0.019;      % 学习率
max_depth = 3;             % 决策树的最大深度
subsample = 0.82;
min_leaf_size = 5;
min_impurity_decrease = 0.01;
test_ratio = 0.4;          % 测试集比例
patience =19;             % 早停机制，容忍连续未改善轮数

% 参数检查
if num_trees <= 0 || max_depth <= 0
    error('参数 num_trees 和 max_depth 必须为正数！');
end

% 调用训练函数
[model, metrics_train, metrics_test] = simpleGBDT(X, y, num_trees, learning_rate, max_depth, test_ratio, patience);

% 输出评价指标
disp('训练集评价指标:');
disp(metrics_train);
disp('测试集评价指标:');
disp(metrics_test);

%% 可视化结果
%figure;

% 测试集真实值与预测值散点图
%subplot(1, 2, 1);
%scatter(metrics_test.y_test, metrics_test.F_test, 25, 'b', 'filled'); hold on;
%plot([min(metrics_test.y_test), max(metrics_test.y_test)], ...
    % [min(metrics_test.y_test), max(metrics_test.y_test)], 'r--', 'LineWidth', 1.5);
%xlabel('真实值');
%ylabel('预测值');
%title('测试集预测值与真实值散点图');
%grid on;

% 测试集真实值与预测值直方图对比
%subplot(1, 2, 2);
%histogram(metrics_test.y_test, 'FaceColor', 'b', 'EdgeColor', 'none', 'Normalization', 'probability'); hold on;
%histogram(metrics_test.F_test, 'FaceColor', 'r', 'EdgeColor', 'none', 'Normalization', 'probability');
%legend('真实值', '预测值');
%xlabel('值');
%ylabel('概率密度');
%title('测试集真实值与预测值直方图对比');
%grid on;

% 残差分析
%figure;
%residuals = metrics_test.y_test - metrics_test.F_test;
%plot(residuals, 'ko', 'MarkerSize', 3);
%title('测试集残差分布');
%xlabel('样本编号');
%ylabel('残差值');
%grid on;

%% 函数定义部分
function [model, metrics_train, metrics_test] = simpleGBDT(X, y, num_trees, learning_rate, max_depth, test_ratio, patience)
    % 数据归一化
    minX = min(X, [], 1);
    maxX = max(X, [], 1);
    X_norm = (X - minX) ./ (maxX - minX);  % 归一化特征

    % 随机划分训练集和测试集
    rng(3111)
    n = size(X, 1);                       % 样本数
    idx = randperm(n);                    % 随机打乱样本索引
    X_norm = X_norm(idx, :);
    y = y(idx);
    split_index = ceil((1 - test_ratio) * n);
    X_train = X_norm(1:split_index, :);   % 训练集特征
    y_train = y(1:split_index);           % 训练集目标值
    X_test = X_norm(split_index+1:end, :); % 测试集特征
    y_test = y(split_index+1:end);        % 测试集目标值

    % 初始化预测值（训练集均值）
    F_train = mean(y_train) * ones(size(y_train));
    F_test = mean(y_train) * ones(size(y_test));

    % 模型存储
    model.trees = cell(num_trees, 1);

    % 梯度提升训练
    best_rmse_test = Inf;  % 保存测试集最小误差
    patience_counter = 0;  % 早停计数器

    for t = 1:num_trees
        % 计算残差
        residual = y_train - F_train;

        % 训练一棵拟合残差的决策树
        tree = fitrtree(X_train, residual, 'MaxNumSplits', max_depth);

        % 保存树模型
        model.trees{t} = tree;

        % 更新训练集和测试集的预测值
        F_train = F_train + learning_rate * predict(tree, X_train);
        F_test = F_test + learning_rate * predict(tree, X_test);

        % 计算测试集 RMSE
        RMSE_test = sqrt(mean((y_test - F_test).^2));

        % 打印日志
        if mod(t, 100) == 0
            fprintf('第 %d 棵树, 测试集 RMSE: %.4f\n', t, RMSE_test);
        end

        % 早停检查
        if RMSE_test < best_rmse_test
            best_rmse_test = RMSE_test;
            patience_counter = 0;  % 重置计数器
        else
            patience_counter = patience_counter + 1;
            if patience_counter >= patience
                fprintf('早停触发: 在第 %d 棵树停止训练。\n', t);
                break;
            end
        end
    end

    % 计算训练集评价指标
    RMSE_train = sqrt(mean((y_train - F_train).^2));
    R2_train = 1 - sum((y_train - F_train).^2) / sum((y_train - mean(y_train)).^2);
    RPD_train = std(y_train) / RMSE_train;

    metrics_train = struct('RMSE', RMSE_train, 'R2', R2_train, 'RPD', RPD_train, ...
                       'y_train', y_train, 'F_train', F_train); % 添加训练集数据

    % 计算测试集评价指标
    R2_test = 1 - sum((y_test - F_test).^2) / sum((y_test - mean(y_test)).^2);
    RPD_test = std(y_test) / RMSE_test;

    metrics_test = struct('RMSE', best_rmse_test, 'R2', R2_test, 'RPD', RPD_test, ...
                          'y_test', y_test, 'F_test', F_test);
    writematrix(y_train, 'E:\data\0.001\daoshu_yijie_CWT_7\3.xlsx', 'Sheet', 'Sheet1', 'WriteMode', 'overwrite');
    writematrix(F_train, 'E:\data\0.001\daoshu_yijie_CWT_7\3.xlsx', 'Sheet', 'Sheet2', 'WriteMode', 'overwrite');
    writematrix(y_test, 'E:\data\P0.001\daoshu_yijie_CWT_7\3.xlsx', 'Sheet', 'Sheet3', 'WriteMode', 'overwrite');
    writematrix(F_test, 'E:\data\P0.001\daoshu_yijie_CWT_7\3.xlsx', 'Sheet', 'Sheet4', 'WriteMode', 'overwrite');
end
