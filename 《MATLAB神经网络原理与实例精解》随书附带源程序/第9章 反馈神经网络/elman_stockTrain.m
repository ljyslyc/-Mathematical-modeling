% elman_stockTrain.m
%% ����
close all
clear,clc

%% ��������
load stock1
%   Name        Size             Bytes  Class     Attributes
% 
%   stock1      1x280             2240  double     

% ��һ������
mi=min(stock1);
ma=max(stock1);
stock1=(stock1-mi)/(ma-mi);

% ����ѵ��������������ݣ�ǰ140��άѵ����������140��ά��������
traindata = stock1(1:140);

%% ѵ��
% ����
P=[];
for i=1:140-5
    P=[P;traindata(i:i+4)];
end
P=P';

% �������
T=[traindata(6:140)];

% ����Elman����
threshold=[0 1;0 1;0 1;0 1;0 1];
% net=newelm(threshold,[0,1],[20,1],{'tansig','purelin'});
net=elmannet;
%  ��ʼѵ��
% ���õ�������
net.trainParam.epochs=1000;
% ��ʼ��
net=init(net); 
net=train(net,P,T);

% ����ѵ���õ�����
save stock_net net

%% ʹ��ѵ�����ݲ���һ��
y=sim(net,P);
error=y-T;
mse(error);

fprintf('error= %f\n', error);

T = T*(ma-mi) + mi;
y = y*(ma-mi) + mi;
plot(6:140,T,'b-',6:140,y,'r-');
title('ʹ��ԭʼ���ݲ���');
legend('��ʵֵ','���Խ��');
