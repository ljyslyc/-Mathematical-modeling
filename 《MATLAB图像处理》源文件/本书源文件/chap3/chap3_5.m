close all;                                       %�رյ�ǰ����ͼ�δ���
clear all;                                        %��չ����ռ����
clc;                                             %����
I = imread('coins.png');                      %��ȡͼ����Ϣ
X = grayslice(I,32);                               %���Ҷ�ͼ��ת��Ϊ����ͼ��
set(0,'defaultFigurePosition',[100,100,1000,500]);   %�޸�ͼ��ͼ��λ�õ�Ĭ������
set(0,'defaultFigureColor', [1 1 1]);                %�޸�ͼ�α�����ɫ������
figure,imshow(I);              %��ʾԭͼ��
figure,imshow(X,jet(32));      %jet(M)���൱��colormap����һ��M��3�����飬

