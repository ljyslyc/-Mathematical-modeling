close all;                   %�رյ�ǰ����ͼ�δ���
clear all;                   %��չ����ռ����
clc;                        %����
X=imread('football.jpg');      %��ȡ�ļ���ʽΪ.jpg,�ļ���Ϊfootball��RGBͼ�����Ϣ
I=rgb2gray(X);              %��RGBͼ��ת��Ϊ�Ҷ�ͼ��
set(0,'defaultFigurePosition',[100,100,1000,500]);  %�޸�ͼ��ͼ��λ�õ�Ĭ������
set(0,'defaultFigureColor', [1 1 1]);               %�޸�ͼ�α�����ɫ������
subplot(121),imshow(X);            %��ʾԭRGBͼ��
subplot(122),imshow(I);             %��ʾת����Ҷ�ͼ��


