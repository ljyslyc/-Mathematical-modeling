% xor_newrbe.m
%% ����
clear all
close all
clc

%% ����
% ��������
P = [0,0,1,1;0,1,1,0]
%P =
%
%     0     0     1     1
%     0     1     1     0
% �������
T = [0,1,0,1]
%T =
%
%     0     1     0     1
%% ��������
net=newrbe(P,T);

% ����
out=sim(net,P)
%% ��ͼ
x=0:.2:1;
N=length(x);                         	%  N=6

% XΪ�µ���������
X(1,:)=reshape(repmat(x,N,1),N*N,1);
X(2,:)=repmat(x,1,N);
out2=sim(net,X);
% ����ΪN*N����
out2=reshape(out2,N,N);
% ��ͼ
mesh(x,x,out2);
