close all;									%�رյ�ǰ����ͼ�δ���
clear all;									%��չ����ռ����
clc;										%����
I=imread('pears.png');						%��ȡͼ����Ϣ
BW=im2bw(I,0.5);							%��RGBͼ��ת��Ϊ��ֵͼ��
set(0,'defaultFigurePosition',[100,100,1000,500]);	%�޸�ͼ��ͼ��λ�õ�Ĭ������
set(0,'defaultFigureColor', [1 1 1]);				%�޸�ͼ�α�����ɫ������
figure,
subplot(121),imshow(I);						%��ʾԭͼ��
subplot(122),imshow(BW);					%��ʾת�����ֵͼ��

