close all;                                    %�رյ�ǰ����ͼ�δ���
clear all;                                     %��չ����ռ����
clc;                                         %����
I=imread('tire.tif');                        %��ȡͼ����Ϣ
H=[1 2 1;0 0 0;-1 -2 -1];                       %����subol����
X=filter2(H,I);                               %�ԻҶ�ͼ��G����2���˲���ʵ�ֱ�Ե���
set(0,'defaultFigurePosition',[100,100,1000,500]);	%�޸�ͼ��ͼ��λ�õ�Ĭ������
set(0,'defaultFigureColor', [1 1 1]);				%�޸�ͼ�α�����ɫ������
figure,
subplot(131),imshow(I);
subplot(132),imshow(X,[]),colorbar();                  %��ʾͼ�񣬲������ɫ��
subplot(133),imshow(X,[]),colorbar('east');