close all;                                    %�رյ�ǰ����ͼ�δ���
clear all;                                     %��չ����ռ����
clc;                                         %����
I=imread('lena.bmp');                         %��ȡͼ����Ϣ
set(0,'defaultFigurePosition',[100,100,1000,500]);    %�޸�ͼ��ͼ��λ�õ�Ĭ������
set(0,'defaultFigureColor', [1 1 1])                  %�޸�ͼ�α�����ɫ������
figure,
subplot(221),imshow(I);
subplot(222),image(I);
subplot(223),image([50,200],[50,300],I);
subplot(224),imagesc(I,[60,150]);
