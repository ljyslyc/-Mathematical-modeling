% pnn_test.m
%% ����
close all
clear,clc

%% ��������
rng(2);
a=rand(14,2)*10;					% ѵ�����ݵ�
p=ceil(a)';

disp('��ȷ���');
tc=[3,1,1,2,1,3,2,3,2,3,3,2,2,3];		% ���
disp(tc);

%% ��ѵ�����ݲ���
y=pnn_net(p,tc,p,1);
disp('���Խ����');
disp(y);

