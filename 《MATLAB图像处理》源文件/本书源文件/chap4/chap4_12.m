close all;							%�رյ�ǰ����ͼ�δ��ڣ���չ����ռ��������������ռ����б���
clear all;
clc;
I=imread('office_1.jpg');				%����ͼ��office_1��office_2������ֵ
J=imread('office_2.jpg');
Ip=imdivide(J,I);					%����ͼ�����
K=imdivide(J,0.5);					%ͼ���һ���������
set(0,'defaultFigurePosition',[100,100,1000,500]);%�޸�ͼ��ͼ��λ�õ�Ĭ������
set(0,'defaultFigureColor',[1 1 1])%�޸�ͼ�α�����ɫ������
figure(1);							%������ʾ�ķ�ͼ��
subplot(121); imshow(I);
subplot(122); imshow(J);
figure(2)
subplot(121); imshow(Ip);
subplot(122); imshow(K);
 