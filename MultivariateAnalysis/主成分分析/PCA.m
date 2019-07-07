%% ���ɷַ��� (��ά)
clc, clear

load example_1.txt  %����Ҫ��ǰ����Ϊ�Ա��������һ��Ϊ�����
data = example_1;

[m, n] = size(data); 
num = 3;   % ѡȡ�����ɷֵĸ���

mu = mean(data); 
sigma = std(data);  %��׼��
%z-score��׼����������������A�����ֵ����Сֵδ֪����������г���ȡֵ��Χ����Ⱥ���ݵ����
%��׼����������=��ԭ����-��ֵ��/��׼��
std_data = zscore(data);
b = std_data(: , 1:end-1);     % �ĸ�����x1, x2, x3, x4
r = cov(b);                                % ������Э�������

% ����Э����������PCA
[PC, latent, explained] = pcacov(r);  %�������ɷ�(PC)��Э�������X������ֵ (latent)��ÿ���������������ڹ۲����ܷ�������ռ�İٷ���(explained)
% �µ����ɷ�z1 = PC(1,1)*x1 + PC(2,1)*x2 + PC(3,1)*x3 + PC(4,1)*x4  , z2 = ...
f = repmat(sign(sum(PC)), size(PC, 1), 1);            %sum(PC)��ʾ�Ծ���PC�������
PC = PC .* f;


%1.��ͨ����С���˷��ع�
regress_args_b = [ones(m, 1), b] \ std_data(:, end);   %��׼�����ݵĻع鷽��ϵ��
bzh = mu ./ sigma;
% ԭʼ���ݵĳ�����
ch10 = mu(end) - bzh(1:end-1) * regress_args_b(2:end) * sigma(end);
fr_1 = regress_args_b(2:end); fr_1 = fr_1';
% ԭʼ���ݵ��Ա�����ϵ��
ch1 = fr_1 ./ sigma(1:end-1) * sigma(end);
% ��ʱģ��Ϊ y = ch10 + ch1[1]*x1 + ch1[2] * x2 + ch1[3] * x3 + ch1[4] * x4
% ����������
check1 = sqrt(sum( (data(:, 1:end-1) * ch1' + ch10 - data(:, end)) .^2 ) / (m - n))


%2.���ɷֻع�ģ��
pca_val = b * PC(:, 1:num);
%���ɷ����ݵĻع鷽��ϵ��
regress_args_pca = [ones(m, 1), pca_val] \ std_data(:, end);
beta = PC(:, 1:num) * regress_args_pca(2:num+1);   %��׼�����ݵĻع鷽��ϵ��
% ԭʼ���ݵĳ�����
ch20 = mu(end) - bzh(1:end-1) * beta * sigma(end);
fr_2 = beta';
% ԭʼ���ݵ��Ա�����ϵ��
ch2 = fr_2 ./ sigma(1:end-1) * sigma(end);
% ��ʱģ��Ϊ y = ch20 + ch2[1]*x1 + ch2[2] * x2 + ch2[3] * x3 + ch2[4] * x4
% ����������
check2 = sqrt(sum( (data(:, 1:end-1) * ch2' + ch20 - data(:, end)) .^2 ) / (m - num - 1))
