close all;                          %�رյ�ǰ����ͼ�δ���
clear all;                           %��չ����ռ����
clc;                               %����
load trees;                         %����ͼ���ļ�trees.mat�������еı�������workspace��
[X1,map1]=imread('forest.tif');        %��ȡͼ����Ϣ
set(0,'defaultFigurePosition',[100,100,1000,500]);    %�޸�ͼ��ͼ��λ�õ�Ĭ������
set(0,'defaultFigureColor', [1 1 1])                  %�޸�ͼ�α�����ɫ������
figure,
subplot(1,2,1),subimage(X,map);     %��ͼ�񴰿ڷֳ�1��2���Ӵ��ڣ�������Ӵ�������ʾͼ��X
subplot(1,2,2),subimage(X1,map1);   %���ұ��Ӵ�������ʾͼ��X1
