% curve_filt_newrb_build.m
%% ����
clear all
close all 
clc

%% ����ԭʼ����
x=-9:8;
y=[129,-32,-118,-138,-125,-97,-55,-23,-4,...
    2,1,-31,-72,-121,-142,-174,-155,-77];

%% ���RBF����
P=x;
T=y;
% ��ʱ��ʼ
tic;
% spread = 2
net = newrb(P, T, 0, 2); 
% ��¼���ĵ�ʱ��
time_cost = toc;

% ����õ���RBFģ��net
save curve_filt_newrb_build net
