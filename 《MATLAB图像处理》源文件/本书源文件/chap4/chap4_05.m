close all;%�رյ�ǰ����ͼ�δ��ڣ���չ����ռ��������������ռ����б���
clear all;
clc;
I=imread('flower.tif');%����flowerͼ��
J=imadd(I,30);         %ÿ������ֵ����30
set(0,'defaultFigurePosition',[100,100,1000,500]);%�޸�ͼ��ͼ��λ�õ�Ĭ������
set(0,'defaultFigureColor',[1 1 1])%�޸�ͼ�α�����ɫ������
subplot(121),imshow(I); %��ʾԭͼ��ͼӳ������ͼ��
subplot(122),imshow(J);
