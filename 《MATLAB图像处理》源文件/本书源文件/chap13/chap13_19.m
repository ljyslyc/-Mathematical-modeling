close all;                  %�رյ�ǰ����ͼ�δ��ڣ���չ����ռ��������������ռ����б���
clear all;
clc;
load mask;% ����ͼ�������
A=X;
load bust; 
B=X;
Fus_Method = struct('name','userDEF','param','myfus_FUN'); % �����ںϹ���͵��ú�����
C=wfusmat(A,B,Fus_Method);%����ͼ���ںϷ���
set(0,'defaultFigurePosition',[100,100,1000,500]);%�޸�ͼ��ͼ��λ�õ�Ĭ������
set(0,'defaultFigureColor',[1 1 1])       %�޸�ͼ�α�����ɫ������
figure                 %����ͼ����ʾ���� 
subplot(1,3,1), imshow(uint8(A)), %��ʾ���
subplot(1,3,2), imshow(uint8(B)), 
subplot(1,3,3), imshow(uint8(C)), 