close all;                  %�رյ�ǰ����ͼ�δ��ڣ���չ����ռ��������������ռ����б���
clear all;
clc;
[A,map]=imread('peppers.png');  %����ͼ��
set(0,'defaultFigurePosition',[100,100,1000,500]);%�޸�ͼ��ͼ��λ�õ�Ĭ������
set(0,'defaultFigureColor',[1 1 1])%�޸�ͼ�α�����ɫ������
[I2,rect]=imcrop(A);              %����ͼ�����
subplot(121),imshow(A); %��ʾԭͼ��
rectangle('Position',rect,'LineWidth',2,'EdgeColor','r') %��ʾͼ���������
subplot(122),imshow(I2);   %��ʾ���е�ͼ��             
