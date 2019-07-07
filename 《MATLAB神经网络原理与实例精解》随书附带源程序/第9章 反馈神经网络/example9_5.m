% example9_5.m
% ����
close all
clear,clc
% ԭʼ����
data =[0.4413	0.4707	0.6953	0.8133;...
       0.4379	0.4677	0.6981	0.8002;...
       0.4517	0.4725	0.7006	0.8201;...
       0.4557	0.4790	0.7019	0.8211;...
       0.4601	0.4811	0.7101	0.8298;...
       0.4612	0.4845	0.7188	0.8312;...
       0.4615	0.4891	0.7201	0.8330];


% ѵ��
net=[];
for i=1:4
    P=[data(1:3,i),data(2:4,i),data(3:5,i)];
    T=[data(4,i),data(5,i),data(6,i)];
    
    th1=[0,1;0,1;0,1];
    th2=[0,1];
    net{i}=newelm(th1,th2,[20,1]);      % ����Elman����
    net{i}=init(net{i});                % ��ʼ��
    net{i}=train(net{i},P,T);           % ѵ��
    
    % ����
    test_P{i}=data(4:6,i);
    y(i)=sim(net{i},test_P{i});         % ����
end
fprintf('��ʵֵ:\n');
disp(data(7,:));
fprintf('Ԥ��ֵ:\n');
disp(y);
fprintf('���:\n');
disp((y-data(7,:))./y);
web -broswer http://www.ilovematlab.cn/forum-222-1.html