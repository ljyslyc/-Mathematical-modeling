close all;                  %�رյ�ǰ����ͼ�δ��ڣ���չ����ռ��������������ռ����б���
clear all;
clc;
I=imread('rice.png');        %����ͼ��rice����ֵ��I
J=imread('cameraman.tif');   %����ͼ��cameraman����ֵ��J
K=imadd(I,J);                %��������ͼ��ļӷ�����
set(0,'defaultFigurePosition',[100,100,1000,500]);%�޸�ͼ��ͼ��λ�õ�Ĭ������
set(0,'defaultFigureColor',[1 1 1])%�޸�ͼ�α�����ɫ������
subplot(131),imshow(I); %��ʾrice��cameraman������Ժ��ͼ��
subplot(132),imshow(J);
subplot(133),imshow(K);
