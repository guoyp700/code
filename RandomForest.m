clc
clear
%% ��������
X=xlsread('E:\��ģ����\P0.01\193_shuxue_r.xlsx','daoshu_yijie','A3:UO195');%��������
y=xlsread('E:\��ģ����\P0.01\193_shuxue_r.xlsx','R','XT2:XT194');%��ѧֵ
%% ���ݼ�����
rng(3111)%������ӣ��������ɭ��ѵ������һ��
R= randperm(193);
n=R(:,1:116);m=R(:,117:193);
% ѵ��������
train_data = X(n,:);
train_labels = y(n,:);
% ���Լ�����
test_data = X(m,:);
test_labels= y(m,:);
%% �������ݹ�һ��
% ѵ�����Ͳ��Լ����룺�Դ��Ĺ�һ����
[train_in,inputps] = mapminmax(train_data');
train_in = train_in';
test_in = mapminmax('apply',test_data',inputps);
test_in = test_in';
% ���Լ��Ͳ��Լ����
[train_out,outputps] = mapminmax(train_labels');
train_out = train_out';
test_out = mapminmax('apply',test_labels',outputps);
test_out = test_out';

%%  ���ɭ��ģ�Ͳ������ü�ѵ��ģ��
trees =12; % ��������Ŀ
leaf  =4; % ��СҶ����
OOBPrediction = 'on';  % �����ͼ
OOBPredictorImportance = 'on'; % ����������Ҫ��
Method = 'regression';  % ѡ��ع�����
net = TreeBagger(trees, train_in, train_out, 'OOBPredictorImportance', OOBPredictorImportance,...
      'Method', Method, 'OOBPrediction', OOBPrediction, 'minleaf', leaf);
importance = net.OOBPermutedPredictorDeltaError;  % ��Ҫ��

%%  �������
Predict_1 = predict(net, train_in );
Predict_2= predict(net, test_in );

%% �壺����һ��
predict_1= mapminmax('reverse',Predict_1,outputps);%ѵ����Ԥ��
predict_2=mapminmax('reverse',Predict_2,outputps);%���Լ�Ԥ��

%% �����������
num1=length(train_labels);%  ѵ��������
num2=length(test_labels);%  ���Լ�����
% ����ѵ�����ľ���ϵ��R2
train_R2 = (num1* sum(predict_1 .* train_labels ) - sum(predict_1) * sum(train_labels ))^2 / ((num1 * sum((predict_1).^2) - (sum(predict_1))^2) * (num1* sum((train_labels ).^2) - (sum(train_labels))^2)); 
% ������Լ��ľ���ϵ��R2
test_R2= (num2* sum(predict_2 .* test_labels) - sum(predict_2) * sum(test_labels))^2 / ((num2 * sum((predict_2).^2) - (sum(predict_2))^2) * (num2* sum((test_labels).^2) - (sum(test_labels))^2));
% Ԥ��������
RMSEC=sqrt(sum((train_labels  - predict_1).^2)/num1);%  ѵ�������������  
RMSEP=sqrt(sum((test_labels- predict_2).^2)/num2);%  ���Լ���������� 
% RPD
STDEV2=std(predict_2);
RPD2=STDEV2/RMSEP;% ���Լ�RPD
STDEV1=std(predict_1);
RPD1=STDEV2/RMSEC;%ѵ����RPD


%% �ߣ�������
fprintf('ѵ��������ϵ��: %.4f\n', train_R2);
fprintf('ѵ�������������: %.4f\n', RMSEC);
fprintf('Ԥ�⼯����ϵ��: %.4f\n', test_R2);
fprintf('Ԥ�⼯���������: %.4f\n', RMSEP);
fprintf('RPD1: %.4f\n',RPD1);
fprintf('RPD2: %.4f\n',RPD2);

% ����train_R2, RMSEC, test_R2, RMSEP, RPD1, RPD2�Ѿ������
% ��������֯��һ��������
%output_data1 = [train_R2; RMSEC; RPD1; test_R2; RMSEP; RPD2];

% ����һ����Ԫ���飬���ڴ洢�������
%headers = {'Train R2', 'RMSEC', 'RPD1', 'Test R2', 'RMSEP', 'RPD2'};

% �����������
%output_matrix = [headers; num2cell(output_data1')];

% ������д��Excel�ļ�
filename = 'E:\��ģ����\P0.01\193_daoshu_yijie_r_RF.xlsx';  % Excel�ļ���
%xlswrite(filename, output_matrix, 'Sheet1', 'J1'); % ��A1��Ԫ��ʼд������


% �ҳ���еĳ���
max_len = max([length(train_labels), length(test_labels), length(predict_1), length(predict_2)]);

% ��������䵽��ͬ���ȣ��̵Ĳ�����NaN���
train_labels_padded = [train_labels; nan(max_len - length(train_labels), 1)];
predict_1_padded = [predict_1; nan(max_len - length(predict_1), 1)];
test_labels_padded = [test_labels; nan(max_len - length(test_labels), 1)];
predict_2_padded = [predict_2; nan(max_len - length(predict_2), 1)];

% ����Щ�а���˳��ϲ�Ϊһ������
output_data = [train_labels_padded, predict_1_padded, test_labels_padded, predict_2_padded];

% ������д��Excel�ļ�
xlswrite(filename, output_data, 'Sheet2', 'A1'); % A1��д�����ʼ��Ԫ��



% %% �ˣ���ͼ
% figure
% plot(train_labels, predict_1, 'o', 'Color','b', 'MarkerFaceColor','none', 'MarkerSize', 5, 'MarkerEdgeColor', 'b');hold on
% plot(test_labels, predict_2, '*', 'Color','r', 'MarkerFaceColor','r', 'MarkerSize', 5);hold on
% plot([5 ,25], [5,25], 'k-', 'LineWidth', 1)
% legend( 'ѵ���� ��Calibration Set��','���Լ� ��Validation Set��')
% legend('boxoff')
% xlabel('ʵ�⣨actual��' )
% ylabel('Ԥ�⣨Predicted��')
% box off