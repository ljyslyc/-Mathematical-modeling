close all;                  %�رյ�ǰ����ͼ�δ��ڣ���չ����ռ��������������ռ����б���
clear all;
clc;
I=imread('lenna.bmp'); %����ͼ��
a=50;b=50;%����ƽ������
J1=move(I,a,b);%�ƶ�ԭͼ��
a=-50;b=50;%����ƽ������
J2=move(I,a,b);%�ƶ�ԭͼ��
a=50;b=-50;%����ƽ������
J3=move(I,a,b);%�ƶ�ԭͼ��
a=-50;b=-50;%����ƽ������
J4=move(I,a,b);%�ƶ�ԭͼ��
set(0,'defaultFigurePosition',[100,100,1000,500]);%�޸�ͼ��ͼ��λ�õ�Ĭ������
set(0,'defaultFigureColor',[1 1 1])%�޸�ͼ�α�����ɫ������
figure,
subplot(1,2,1),imshow(J1),axis on;%�����ƶ���ͼ��
subplot(1,2,2),imshow(J2),axis on;%�����ƶ���ͼ��
figure,
subplot(1,2,1),imshow(J3),axis on;%�����ƶ���ͼ��
subplot(1,2,2),imshow(J4),axis on;%�����ƶ���ͼ��

