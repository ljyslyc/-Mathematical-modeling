%����11-16��
close all;							%�رյ�ǰ����ͼ�δ��ڣ���չ����ռ��������������ռ����б���
clear all;
clc;
I=imread('number.jpg');                 %����ͼ��
K=im2bw(I);                           %Iת��Ϊ��ֵͼ��
J=~K;                                %ͼ��ȡ��
EUL=bweuler(J)                       %��ͼ���ŷ����
set(0,'defaultFigurePosition',[100,100,1000,500]);  
set(0,'defaultFigureColor',[1 1 1])
figure;subplot(131);imshow(I);           %���ԭͼ
subplot(132);imshow(K);                %��ֵͼ
subplot(133);imshow(J);                %ȡ�����ͼ
