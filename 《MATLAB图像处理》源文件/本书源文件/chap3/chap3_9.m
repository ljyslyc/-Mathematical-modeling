close all;                                       %�رյ�ǰ����ͼ�δ���
clear all;                                        %��չ����ռ����
clc;                                             %����
load trees;                                       %���ļ���trees��mat�����������ݵ�workplace
BW = im2bw(X,map,0.4);                          %����=����ͼ��ת��Ϊ��ֵͼ��
set(0,'defaultFigurePosition',[100,100,1000,500]);   %�޸�ͼ��ͼ��λ�õ�Ĭ������
set(0,'defaultFigureColor', [1 1 1]);                %�޸�ͼ�α�����ɫ������
figure, imshow(X,map);         %��ʾԭ����ͼ��
figure, imshow(BW);           %��ʾת�����ֵͼ��
