close all;                  %�رյ�ǰ����ͼ�δ��ڣ���չ����ռ��������������ռ����б���
clear all;
clc;
I=imread('cameraman.tif'); %����ͼ��
J1=mirror(I,1);%ԭͼ���ˮƽ����
J2=mirror(I,2);%ԭͼ��Ĵ�ֱ����
J3=mirror(I,3);%ԭͼ���ˮƽ��ֱ����
set(0,'defaultFigurePosition',[100,100,1000,500]);%�޸�ͼ��ͼ��λ�õ�Ĭ������
set(0,'defaultFigureColor',[1 1 1])%�޸�ͼ�α�����ɫ������
figure,
subplot(1,2,1),imshow(I) ;%����ԭͼ��
subplot(1,2,2),imshow(J1);%����ˮƽ�����ͼ��
figure,
subplot(1,2,1),imshow(J2);%����ˮƽ�����ͼ��
subplot(1,2,2),imshow(J3);%���ƴ�ֱ�����ͼ��

