close all;                                        %�رյ�ǰ����ͼ�δ���
clear all;                                        %��չ����ռ����
clc;                                             %����
I1=imread('football.jpg');                           %��ȡһ��RGBͼ��
I2=imread('cameraman','tif');                       %��ȡһ���Ҷ�ͼ��
I3=imread('E:\onion.png');                         %��ȡ�ǵ�ǰ·���µ�һ��RGBͼ��
set(0,'defaultFigurePosition',[100,100,1000,500]);    %�޸�ͼ��ͼ��λ�õ�Ĭ������
set(0,'defaultFigureColor', [1 1 1])                  %�޸�ͼ�α�����ɫ������
figure,
subplot(1,3,1),imshow(I1);              %��ʾ��RGBͼ��
subplot(1,3,2),imshow(I2);               %��ʾ�ûҶ�ͼ��
subplot(1,3,3),imshow(I3);               %��ʾ��RGBͼ��  
