close all;                          %�رյ�ǰ����ͼ�δ���
clear all                           %��չ����ռ����
clc;                               %����
set(0,'defaultFigurePosition',[100,100,1000,500]);    %�޸�ͼ��ͼ��λ�õ�Ĭ������
set(0,'defaultFigureColor', [1 1 1])                  %�޸�ͼ�α�����ɫ������
h = imshow('hestain.png');                        %��ʾͼ��
hp = impixelinfo;                                 %����ͼ��������Ϣ��ʾ����
set(hp,'Position',[150 290 300 20]);                 %����������Ϣ������ʾ��λ��
figure
imshow('trees.tif');
impixelinfo                                 %����ͼ��������Ϣ��ʾ����
