close all;                  %�رյ�ǰ����ͼ�δ��ڣ���չ����ռ��������������ռ����б���
clear all;
clc;
A=imread('office_2.jpg');                   %����ͼ��
J1=imrotate(A, 30);                         %������ת�Ƕȣ�ʵ����ת����ʾ
J2=imrotate(A, -30);
J3=imrotate(A,30,'bicubic','crop');        %�������ͼ���С��ʵ����תͼ����ʾ 
J4=imrotate(A,30, 'bicubic','loose'); 
set(0,'defaultFigurePosition',[100,100,1000,500]);%�޸�ͼ��ͼ��λ�õ�Ĭ������
set(0,'defaultFigureColor',[1 1 1])%�޸�ͼ�α�����ɫ������
figure(1)                                  %��ʾ��ת������
subplot(121),imshow(J1);
subplot(122),imshow(J2);  
figure(2)
subplot(121),imshow(J3);
subplot(122),imshow(J4);
