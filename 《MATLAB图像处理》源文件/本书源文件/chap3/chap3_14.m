close all;                                    %�رյ�ǰ����ͼ�δ���
clear all;                                     %��չ����ռ����
clc;                                         %����
I=imread('lena.bmp');                         %��ȡͼ����Ϣ
set(0,'defaultFigurePosition',[100,100,1000,500]);    %�޸�ͼ��ͼ��λ�õ�Ĭ������
set(0,'defaultFigureColor', [1 1 1])                  %�޸�ͼ�α�����ɫ������
subplot(121),imshow(I,128);                         %��128�Ҷȼ���ʾ�ûҶ�ͼ��
subplot(122),imshow(I,[60,120]);                     %���ûҶ�����Ϊ[60,120]��ʾ�ûҶ�ͼ��

