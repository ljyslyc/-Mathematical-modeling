% credit_class.m
% �Ŵ����õ�����
% ����ȡ�Ե¹��������ݿ�

%% �������ռ�
clear,clc

% �ر�ͼ�δ���
close all

%% ��������
% ���ļ�
fid = fopen('german.data', 'r');

% ����ʽ��ȡÿһ��
% ÿ�а���21������ַ���������
C = textscan(fid, '%s %d %s %s %d %s %s %d %s %s %d %s %d %s %s %d %s %d %s %s %d\n');

% �ر��ļ�
fclose(fid);

% ���ַ���ת��Ϊ����
N = 20;
% ���������������ֵ����
C1=zeros(N+1,1000);
for i=1:N+1
    % �������
    if iscell(C{i})
        for j=1:1000
            % eg: 'A12' -> 2
            if i<10
                d = textscan(C{i}{j}, '%c%c%d');
            % eg: 'A103'  -> 3
            else
                d = textscan(C{i}{j}, '%c%c%c%d');
            end
            C1(i,j) = d{end};
        end
    % ��ֵ����
    else
        C1(i,:) = C{i};
    end
end

%% ����ѵ���������������

% ��������
x = C1(1:N, :);
% Ŀ�����
y = C1(N+1, :);

% ����
posx = x(:,y==1);
% ����
negx = x(:,y==2);

% ѵ������
trainx = [ posx(:,1:350), negx(:,1:150)];
trainy = [ones(1,350), ones(1,150)*2];

% ��������
testx = [ posx(:,351:700), negx(:,151:300)];
testy = trainy;
%% ������һ��
% ѵ��������һ��
[trainx, s1] = mapminmax(trainx);

% ����������һ��
testx = mapminmax('apply', testx, s1);
%% �������磬ѵ��

% ����BP����
net = newff(trainx, trainy);
% �������ѵ������
net.trainParam.epochs = 1500;
% Ŀ�����
net.trainParam.goal = 1e-13;
% ��ʾ����
net.trainParam.show = 1;

% ѵ��
net = train(net,trainx, trainy);
%% ����
y0 = net(testx);

% y0Ϊ�������������y0����Ϊ1��2��
y00 = y0;
% ��1.5Ϊ�ٽ�㣬С��1.5Ϊ1������1.5Ϊ2
y00(y00<1.5)=1;
y00(y00>1.5)=2;

% ��ʾ��ȷ��
fprintf('��ȷ��: \n');
disp(sum(y00==testy)/length(y00));
web -broswer http://www.ilovematlab.cn/forum-222-1.html