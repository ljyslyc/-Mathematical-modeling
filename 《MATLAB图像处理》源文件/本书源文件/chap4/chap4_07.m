close all;                          %�رյ�ǰ����ͼ�δ��ڣ���չ����ռ��������������ռ����б���
clear all;
clc;
RGB=imread('eight.tif');            %����eightͼ�񣬸�ֵ��RGB
M1=3;
[BW1,runningt1]=Denoise(RGB,M1); % M=3����
M2=9;
[BW2,runningt2]=Denoise(RGB,M2); % M=9����
set(0,'defaultFigurePosition',[100,100,1000,500]);%�޸�ͼ��ͼ��λ�õ�Ĭ������
set(0,'defaultFigureColor',[1 1 1])%�޸�ͼ�α�����ɫ������
subplot(121); imshow(BW1);   %��ʾ���
subplot(122); imshow(BW2); 
disp('����4������ʱ��')
runningt1
disp('����10������ʱ��')
runningt2
