close all;                  %�رյ�ǰ����ͼ�δ��ڣ���չ����ռ��������������ռ����б���
clear all;
clc;
X=imread('girl.bmp');      %��ȡͼ��
X=rgb2gray(X);             %ת��ͼ����������
[ca1,chd1,cvd1,cdd1] = dwt2(X,'bior3.7');
set(0,'defaultFigurePosition',[100,100,1000,500]);%�޸�ͼ��ͼ��λ�õ�Ĭ������
set(0,'defaultFigureColor',[1 1 1])  
figure                             %��ʾС���任��������
subplot(141); 
imshow(uint8(ca1));
subplot(1,4,2); 
imshow(chd1);
subplot(1,4,3); 
imshow(cvd1);
subplot(1,4,4); 
imshow(cdd1);                      %��ʾԭͼ��С���任�������ͼ��
figure
subplot(121),imshow(X);          
subplot(122),imshow([ca1,chd1;cvd1,cdd1]);