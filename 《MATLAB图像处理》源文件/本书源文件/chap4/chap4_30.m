close all;                  %�رյ�ǰ����ͼ�δ��ڣ���չ����ռ��������������ռ����б���
clear all;
clc;
I=imread('pout.tif');           %����ԭͼ��
BW1=roicolor(I,55,100);                   %���ڻҶ�ͼ��ROI����ѡȡ
c=[87 171 201 165 79 32 87];
r=[133 133 205 259 259 209 133];%����ROI����λ��
BW=roipoly(I,c,r); %����c��rѡ��ROI����
I1=roifill(I,BW); %��������BW��Ĥͼ������������
h=fspecial('motion',20,45); %����motion�˲�����˵������
I2=roifilt2(h,I,BW); %���������˲�
set(0,'defaultFigurePosition',[100,100,1000,500]);%�޸�ͼ��ͼ��λ�õ�Ĭ������
set(0,'defaultFigureColor',[1 1 1])%�޸�ͼ�α�����ɫ������
figure
subplot(121),imshow(BW1); %��ʾ������
subplot(122),imshow(BW); %��ʾROI����
figure
subplot(121),imshow(I1);%��ʾ���Ч��
subplot(122),imshow(I2); %��ʾ�����˲�Ч��
 


