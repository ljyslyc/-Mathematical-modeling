%����11-4����HSV�ռ��ֱ��ͼ��δ��H,S,V����������
close all;							%�رյ�ǰ����ͼ�δ��ڣ���չ����ռ��������������ռ����б���
clear all;
clc;
J=imread('huangguahua.jpg');				%����Ҫ�����ͼ�񣬲���ֵ��J
hsv = rgb2hsv(J);                   %ͼ����RGB�ռ�任��HSV�ռ�
h = hsv(:, :, 1);                     %Ϊɫ��h��ֵ
s = hsv(:, :, 2);                     %Ϊ���Ͷ�s��ֵ
v = hsv(:, :, 3);                     %Ϊ����v��ֵ
set(0,'defaultFigurePosition',[100,100,1000,500]);	%�޸�ͼ��ͼ��λ�õ�Ĭ������
set(0,'defaultFigureColor',[1 1 1])
figure;subplot(121);imshow(J);                           %��ʾԭͼ
subplot(122);imshow(h);         %����ɫ��h�ĻҶ�ͼ��
figure;subplot(121);imshow(s);   %���ڱ��Ͷ�s�ĻҶ�ͼ��
subplot(122);imshow(v);         %��������v�ĻҶ�ͼ��
figure;subplot(131);imhist(h); 	      	%��ʾɫ��h��ֱ��ͼ
subplot(132);imhist(s);              %��ʾ���Ͷ�s��ֱ��ͼ
subplot(133);imhist(v);              %��ʾ����v��ͼ



