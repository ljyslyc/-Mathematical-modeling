close all;                  			%�رյ�ǰ����ͼ�δ��ڣ���չ����ռ��������������ռ����б���
clear all;
clc;
I=imread('trees.tif'); 					%����ͼ��
J1=transp(I);						%��ԭͼ���ת��
I1=imread('lenna.bmp'); 				%����ͼ��
J2=transp(I1);						%��ԭͼ���ת��
set(0,'defaultFigurePosition',[100,100,1000,500]);%�޸�ͼ��ͼ��λ�õ�Ĭ������
set(0,'defaultFigureColor',[1 1 1])%�޸�ͼ�α�����ɫ������
figure,
subplot(1,2,1),imshow(J1);%�����ƶ���ͼ��
subplot(1,2,2),imshow(J2);%�����ƶ���ͼ��
