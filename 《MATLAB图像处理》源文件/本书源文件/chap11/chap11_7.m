%����11.7����ͼ��ĻҶȹ�������
close all;							    %�رյ�ǰ����ͼ�δ��ڣ���չ����ռ��������������ռ����б���
clear all;
clc;
I = imread('circuit.tif');              %����ͼ��circuit.tif
glcm = graycomatrix(I,'Offset',[0 2]);  %ͼ��I�ĻҶȹ�������2��ʾ��ǰ�������ھӵľ���Ϊ2��offsetΪ[0 2]��ʾ�Ƕ�Ϊ0Ϊˮƽ����
set(0,'defaultFigurePosition',[100,100,1000,500]);	%�޸�ͼ��ͼ��λ�õ�Ĭ������
set(0,'defaultFigureColor',[1 1 1])
imshow(I);
glcm
