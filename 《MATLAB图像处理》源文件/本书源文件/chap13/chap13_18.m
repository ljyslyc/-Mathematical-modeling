close all;                  %�رյ�ǰ����ͼ�δ��ڣ���չ����ռ��������������ռ����б���
clear all;
clc;
X1 = imread('girl.bmp');          % ����ԭʼ����ͼ��
X2 = imread('lenna.bmp');
FUSmean = wfusimg(X1,X2,'db2',5,'mean','mean');%ͨ������wfusingʵ������ͼ���ں�
FUSmaxmin = wfusimg(X1,X2,'db2',5,'max','min');
set(0,'defaultFigurePosition',[100,100,1000,500]);%�޸�ͼ��ͼ��λ�õ�Ĭ������
set(0,'defaultFigureColor',[1 1 1])       %�޸�ͼ�α�����ɫ������
figure                 %����ͼ����ʾ����
subplot(121), imshow(uint8(FUSmean))
subplot(122), imshow(uint8(FUSmaxmin))