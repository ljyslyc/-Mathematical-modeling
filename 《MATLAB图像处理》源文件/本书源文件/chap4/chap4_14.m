close all;							%�رյ�ǰ����ͼ�δ��ڣ���չ����ռ��������������ռ����б���
clear all;
clc;
J=imread('rice.png');% ��ȡ�Ҷ�ͼ�񣬸�ֵ��J
J1=im2bw(J);%���Ҷ�ͼ��ת���ɶ�ֵͼ�񣬸�ֵ��J1
J2=imcomplement(J);%��Ҷ�ͼ��Ĳ�����ֵ��J2
J3=imcomplement(J1);%���ֵͼ��Ĳ�����ֵ��J3
set(0,'defaultFigurePosition',[100,100,1000,500]);%�޸�ͼ��ͼ��λ�õ�Ĭ������
set(0,'defaultFigureColor',[1 1 1])         %�޸�ͼ�α�����ɫ������
figure,                              %��ʾ������
subplot(131),imshow(J1)             %��ʾ�Ҷ�ͼ���䲹ͼ��
subplot(132),imshow(J2)         %��ʾ��ֵͼ���䲹ͼ��
subplot(133),imshow(J3) 