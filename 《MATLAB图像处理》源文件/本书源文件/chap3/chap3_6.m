close all;									%�رյ�ǰ����ͼ�δ���
clear all;									%��չ����ռ����
clc;										%����
[X,map]=imread('forest.tif');%����Ϣ
I = ind2gray(X,map);							%�ٽ�����ͼ��ת��Ϊ�Ҷ�ͼ��
set(0,'defaultFigurePosition',[100,100,1000,500]);	%�޸�ͼ��ͼ��λ�õ�Ĭ������
set(0,'defaultFigureColor', [1 1 1]);				%�޸�ͼ�α�����ɫ������
figure,imshow(X,map);					%������ͼ����ʾ
figure,imshow(I);						%���Ҷ�ͼ����ʾ

