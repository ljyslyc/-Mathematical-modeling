close all;                  %�رյ�ǰ����ͼ�δ��ڣ���չ����ռ��������������ռ����б���
clear all;
clc;
[A,map]=imread('peppers.png');  %����ͼ��
rect=[75 68 130 112];           %�����������
X1=imcrop(A,rect);              %����ͼ�����
set(0,'defaultFigurePosition',[100,100,1000,500]);%�޸�ͼ��ͼ��λ�õ�Ĭ������
set(0,'defaultFigureColor',[1 1 1])%�޸�ͼ�α�����ɫ������
subplot(121),imshow(A); %��ʾԭͼ��
rectangle('Position',rect,'LineWidth',2,'EdgeColor','r') %��ʾͼ���������
subplot(122),imshow(X1);   %��ʾ���е�ͼ��             
