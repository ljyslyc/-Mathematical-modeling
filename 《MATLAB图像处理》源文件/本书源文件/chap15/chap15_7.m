clear all;  			%��������ռ䣬�ر�ͼ�δ��ڣ����������
close all;
clc;
B=imread('girl2.bmp');%����ͼ��
C=imread('boy1.bmp');
BW1=face_detection(B);%���ú���face_detection����������� 
BW2=face_detection(C);
set(0,'defaultFigurePosition',[100,100,1200,450]);%�޸�ͼ��ͼ��λ�õ�Ĭ������
set(0,'defaultFigureColor',[1 1 1])       %�޸�ͼ�α�����ɫ������ 
figure,
subplot(121),imshow(B);%��ʾԭͼ�����
subplot(122),imshow(BW1);
figure,
subplot(121),imshow(C);
subplot(122),imshow(BW2);