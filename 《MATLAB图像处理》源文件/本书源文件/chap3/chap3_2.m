close all;                                      %�رյ�ǰ����ͼ�δ���
clear all;                                      %��չ����ռ����
clc;                                            %����
[X,map] = imread('trees.tif');                      %��ȡԭͼ����Ϣ
newmap = rgb2gray(map);                             %����ɫ��ɫӳ���ת��Ϊ�Ҷ���ɫӳ���
set(0,'defaultFigurePosition',[100,100,1000,500]);  %�޸�ͼ��ͼ��λ�õ�Ĭ������
set(0,'defaultFigureColor', [1 1 1]);               %�޸�ͼ�α�����ɫ������
figure,imshow(X,map);                   %��ʾԭͼ��
figure,imshow(X,newmap);                %��ʾת����Ҷ�ͼ��
