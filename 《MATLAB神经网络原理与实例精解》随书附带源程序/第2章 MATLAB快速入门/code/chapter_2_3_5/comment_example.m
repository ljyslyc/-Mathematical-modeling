% comment_example.m
%% 
clear,close all;
clc

%% ���ļ�
fid=fopen('data.txt','rb');

%% ��ȡ����
data=fread(fid, 10, 'uint8');    % ��ȡ10������
d=data.^2;

%{
plot(data,d);
title('ɢ��ͼ');
xlabel('x');
ylabel('y');
%}
%% �ر��ļ�
fclose(fid);