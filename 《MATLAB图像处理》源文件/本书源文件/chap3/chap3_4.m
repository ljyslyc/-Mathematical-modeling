close all                                         %�رյ�ǰ����ͼ�δ���
clear all;                                         %��չ����ռ����
clc                                              %����
I = imread('cameraman.tif')                         %��ȡ�Ҷ�ͼ����Ϣ                        
[X,map]=gray2ind(I,8);                          %ʵ�ֻҶ�ͼ��������ͼ���ת��,Nȡ8
set(0,'defaultFigurePosition',[100,100,1000,500]);  %�޸�ͼ��ͼ��λ�õ�Ĭ������
set(0,'defaultFigureColor', [1 1 1]);               %�޸�ͼ�α�����ɫ������
figure,imshow(I);              %��ʾԭ�Ҷ�ͼ��
figure, imshow(X, map);       %��ʾN=8ת��������ͼ��



