close all; clear all; clc;			%�ر�����ͼ�δ��ڣ���������ռ����б��������������
y=randn(1000,1);				%������̬�ֲ�������
figure;
subplot(121);hist(y);				%����hist����Ĭ��ֱ��ͼ
subplot(122);hist(y,20)			%����hist����ָ��ֱ��ͼ
