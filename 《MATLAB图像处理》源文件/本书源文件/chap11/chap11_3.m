%����11-3��
close all;							%�رյ�ǰ����ͼ�δ��ڣ���չ����ռ��������������ռ����б���
clear all;
clc;
I=imread('huangguahua.jpg');      %����Ҫ�����ͼ�񣬲���ֵ��I
R=I(:,:,1);                         %ͼ���R����
G=I(:,:,2);                         %ͼ���G����
B=I(:,:,3);                         %ͼ���B����
set(0,'defaultFigurePosition',[100,100,1000,500]);	%�޸�ͼ��ͼ��λ�õ�Ĭ������
set(0,'defaultFigureColor',[1 1 1]) 
figure;subplot(121);imshow(I);                            %��ʾ��ɫͼ��
subplot(122);imshow(R);          %R�����Ҷ�ͼ
figure;subplot(121);imshow(G);          %G�����Ҷ�ͼ
subplot(122);imshow(B);          %B�����Ҷ�ͼ
figure;subplot(131);
imhist(I(:,:,1))              %��ʾ��ɫ�ֱ����µ�ֱ��ͼ
subplot(132);imhist(I(:,:,2))              %��ʾ��ɫ�ֱ����µ�ֱ��ͼ
subplot(133);imhist(I(:,:,3))  %��ʾ��ɫ�ֱ����µ�ֱ��ͼ
