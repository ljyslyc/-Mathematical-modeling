close all;              %�رյ�ǰ����ͼ�δ��ڣ���չ����ռ��������������ռ����б���
clear all;
clc
A=imread('ipexroundness_04.png');%����ԭʼͼ��ֵ��A��B
B=imread('ipexroundness_01.png');
C=immultiply(A,B);              %����A��B�ĳ˷������������ظ�C             
A1=im2double(A);                %��A��Bת����˫�������ͣ���ΪA1��B1
B1=im2double(B);
C1=immultiply(A1,B1);           %���¼���A1��B1�ĳ˻���������ظ�C1
set(0,'defaultFigurePosition',[100,100,1000,500]);%�޸�ͼ��ͼ��λ�õ�Ĭ������
set(0,'defaultFigureColor',[1 1 1])%�޸�ͼ�α�����ɫ������
figure(1),% ��ʾԭͼ��A��B
subplot(121),imshow(A),axis on;
subplot(122),imshow(B),axis on;
figure(2),% ��ʾuint8��doubleͼ�����ݸ�ʽ�£��˻�C��C1
subplot(121),imshow(C),axis on;;
subplot(122),imshow(C1),axis on;;



