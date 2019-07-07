clear all;  			%��������ռ䣬�ر�ͼ�δ��ڣ����������
close all;
clc;
I=imread('girl1.bmp');
I1=refine_face_detection(I); 			%�����ָ�
[m,n]=size(I1);
theta1=0;							%����
theta2=pi/2;
f = 0.88;							%����Ƶ��
sigma = 2.6;						%����
Sx = 5;
Sy = 5;							%����Ⱥͳ���
Gabor1=Gabor_hy(Sx,Sy,f,theta1,sigma);%����Gabor�任�Ĵ��ں���
Gabor2=Gabor_hy(Sx,Sy,f,theta2,sigma);%����Gabor�任�Ĵ��ں���
Regabout1=conv2(I1,double(real(Gabor1)),'same');
Regabout2=conv2(I1,double(real(Gabor2)),'same');
set(0,'defaultFigurePosition',[100,100,1200,450]); %�޸�ͼ��ͼ��λ�õ�Ĭ������
set(0,'defaultFigureColor',[1 1 1])                 %�޸�ͼ�α�����ɫ������
figure,
subplot(131),imshow(I);
subplot(132),imshow(Regabout1);
subplot(133),imshow(Regabout2);
