close all;									%�رյ�ǰ����ͼ�δ���
clear all;									%��չ����ռ����
clc;										%����
I=imread('rice.png');							%��ȡͼ����Ϣ
BW1=im2bw(I,0.4);							%���Ҷ�ͼ��ת��Ϊ��ֵͼ��levelֵΪ0.4			
BW2=im2bw(I,0.6);							%���Ҷ�ͼ��ת��Ϊ��ֵͼ��levelֵΪ0.6
set(0,'defaultFigurePosition',[100,100,1000,500]);	%�޸�ͼ��ͼ��λ�õ�Ĭ������
set(0,'defaultFigureColor', [1 1 1]);				%�޸�ͼ�α�����ɫ������
figure;
subplot(131),imshow(I);					%��ʾlevel=0.4ת����Ķ�ֵͼ��
subplot(132),imshow(BW1);					%��ʾlevel=0.5ת����Ķ�ֵͼ��
subplot(133),imshow(BW2);					%��ʾlevel=0.6ת����Ķ�ֵͼ��
