close all;							%�رյ�ǰ����ͼ�δ��ڣ���չ����ռ��������������ռ����б���
clear all;
clc;
I=imread('cameraman.tif'); %��ȡͼ��
J=imread('rice.png');
K1=imlincomb(1.0,I,1.0,J); %����ͼ�����
K2=imlincomb(1.0,I,-1.0,J,'double'); %����ͼ�����
K3=imlincomb(2,I); %ͼ��ĳ˷�
K4=imlincomb(0.5,I);%ͼ��ĳ���
set(0,'defaultFigurePosition',[100,100,1000,500]);%�޸�ͼ��ͼ��λ�õ�Ĭ������
set(0,'defaultFigureColor',[1 1 1])%�޸�ͼ�α�����ɫ������
figure(1),            %��ʾ���
subplot(121),imshow(K1);
subplot(122),imshow(K2);
figure,
subplot(121),imshow(K3);
subplot(122),imshow(K4);