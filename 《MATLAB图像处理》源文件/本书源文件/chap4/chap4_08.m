clear all;          %�رյ�ǰ����ͼ�δ��ڣ���չ����ռ��������������ռ����б���
clc
close all;
A=imread('cameraman.tif');%
B=imread('testpat1.png');
C=imsubtract(A,B);  %����ͼ�����
set(0,'defaultFigurePosition',[100,100,1000,500]);%�޸�ͼ��ͼ��λ�õ�Ĭ������
set(0,'defaultFigureColor',[1 1 1])%�޸�ͼ�α�����ɫ������
figure,%��ʾԭͼ�񼰲���ͼ��
subplot(121),imshow(C);
subplot(122),imshow(255-C);