close all;                  %�رյ�ǰ����ͼ�δ��ڣ���չ����ռ��������������ռ����б���
clear all;
clc;
I=imread('tire.tif');           %����ͼ��
I1=im2double(I);                %��ֵ����ת��
f=@(x) min(x);
I2=colfilt(I1,[4 4],'sliding',f);    %���ջ�������ʽ ���ж�ͼ�������Сֵ�������
m=2;n=2;
f=@(x) ones(m*n,1)*min(x);                    %
I3=colfilt(I1,[m n],'distinct',f);
m=4;n=4;
f=@(x) ones(m*n,1)*min(x);
I4=colfilt(I1,[4 4],'distinct',f);
set(0,'defaultFigurePosition',[100,100,1000,500]);%�޸�ͼ��ͼ��λ�õ�Ĭ������
set(0,'defaultFigureColor',[1 1 1])%�޸�ͼ�α�����ɫ������
figure
subplot(131),imshow(I2);
subplot(132),imshow(I3);       %��ʾԭͼ��ʹ����ͼ��
subplot(133),imshow(I4);  