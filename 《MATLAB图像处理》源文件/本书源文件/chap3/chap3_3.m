close all;									%�رյ�ǰ����ͼ�δ���
clear all;									%��չ����ռ����
clc;										%����
RGB = imread('football.jpg');					%��ȡͼ����Ϣ
[X1,map1]=rgb2ind(RGB,64);					%��RGBͼ��ת��������ͼ����ɫ����N��64��      
[X2,map2]=rgb2ind(RGB,0.2);					%��RGBͼ��ת��������ͼ����ɫ����N��216��
map3= colorcube(128);						%����һ��ָ����ɫ��Ŀ��RGB��ɫӳ���
X3=rgb2ind(RGB,map3);
set(0,'defaultFigurePosition',[100,100,1000,500]); 	%�޸�ͼ��ͼ��λ�õ�Ĭ������
set(0,'defaultFigureColor', [1 1 1]);				%�޸�ͼ�α�����ɫ������
figure; 
subplot(131),imshow(X1,map1); %��ʾ����С���ת��������ͼ��
subplot(132),imshow(X2,map2); %��ʾ�þ���������ת��������ͼ��
subplot(133),imshow(X3,map3); %��ʾ����ɫ���Ʒ�ת��������ͼ��



