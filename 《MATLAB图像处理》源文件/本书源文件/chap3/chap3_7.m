close all;                                      %�رյ�ǰ����ͼ�δ���
clear all                                       %��չ����ռ����
clc;                                           %����
[X,map]=imread('kids.tif');                        %��ȡͼ����Ϣ
RGB=ind2rgb(X,map);                      %������ͼ��ת��Ϊ���ɫͼ��
set(0,'defaultFigurePosition',[100,100,1000,500]);  %�޸�ͼ��ͼ��λ�õ�Ĭ������
set(0,'defaultFigureColor', [1 1 1]);               %�޸�ͼ�α�����ɫ������
figure, imshow(X,map);						%��ʾԭͼ��
figure,imshow(RGB);							%��ʾ���ɫͼ��


