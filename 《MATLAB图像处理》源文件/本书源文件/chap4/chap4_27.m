close all;                  %�رյ�ǰ����ͼ�δ��ڣ���չ����ռ��������������ռ����б���
clear all;
clc;
A = imread('cameraman.tif'); %��ȡͼ��
A1=im2double(A);            %��ֵ����ת��
B1 = nlfilter(A1,[4 4],'std2');% ��ͼ��A���û�����������������д���
fun = @(x) max(x(:));           % ��ͼ��A���û�����������������д���
B2 = nlfilter(A1,[3 3],fun);
B3 = nlfilter(A1,[6 6],fun);
set(0,'defaultFigurePosition',[100,100,1000,500]);%�޸�ͼ��ͼ��λ�õ�Ĭ������
set(0,'defaultFigureColor',[1 1 1])%�޸�ͼ�α�����ɫ������
subplot(131),imshow(B1); %��ʾ�����ͼ��
subplot(132),imshow(B2);      
subplot(133),imshow(B3);  