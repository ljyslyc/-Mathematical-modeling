close all; 							%�رյ�ǰ����ͼ�δ��ڣ���չ����ռ��������������ռ����б���
clear all;
clc
A=imread('house.jpg');				%����ͼ�񣬸�ֵ��A
B=immultiply(A,1.5);					%�ֱ������������1.5��0.5��������ظ�B��C
C=immultiply(A,0.5);
set(0,'defaultFigurePosition',[100,100,1000,500]);%�޸�ͼ��ͼ��λ�õ�Ĭ������
set(0,'defaultFigureColor',[1 1 1])%�޸�ͼ�α�����ɫ������
figure(1),
subplot(1,2,1),imshow(B),axis on;%��ʾ�������������Ժ��ͼ��
subplot(1,2,2),imshow(C),axis on;

