close all;%�رյ�ǰ����ͼ�δ��ڣ���չ����ռ��������������ռ����б���
clc;
clear all;
I=imread('ipexroundness_01.png');%����ͼ�񣬸�ֵ��I��J
J=imread('ipexroundness_04.png');
I1=im2bw(I);                    %ת��Ϊ��ֵͼ��
J1=im2bw(J);
K1=I1 & J1;                     %ʵ��ͼ����߼����롱����
K2=I1 | J1;                     %ʵ��ͼ����߼���������
K3=~I1;                         %ʵ���߼����ǡ�����
K4=xor(I1,J1);                  %ʵ�֡��������
set(0,'defaultFigurePosition',[100,100,1000,500]);%�޸�ͼ��ͼ��λ�õ�Ĭ������
set(0,'defaultFigureColor',[1 1 1])%�޸�ͼ�α�����ɫ������
figure,                     %��ʾԭͼ����Ӧ�Ķ�ֵͼ�� 
subplot(121);imshow(I1),axis on; 
subplot(122);imshow(J1),axis on; 
figure,                      %��ʾ�߼�����ͼ��
subplot(121);imshow(K1),axis on; 
subplot(122);imshow(K2),axis on;
figure, 
subplot(121);imshow(K3),axis on;
subplot(122);imshow(K4),axis on;
