%curve_filt_hand_sim.m
%% ����
clear all
close all
clc
%% ����ģ��
load net.mat

%% ����
%����
xx=-9:.2:8;

% �������뵽���ĵľ���
t = x;
zz=dist(xx',t);

% ��������������
 GG=radbas(zz);

% �������������
Y=GG*w;

%% ��ͼ
% ԭʼ���ݵ�
plot(x,y,'o');
hold on;
% ��ϵĺ�������
plot(xx,Y','-');
legend('ԭʼ����','�������');
title('�þ���������������');
