close all;                    %�رյ�ǰ����ͼ�δ���
clear all;                     %��չ����ռ����
clc;                         %����
%I=imread('testpat.png'); 
I=imread('football.jpg');     %��ȡͼ����Ϣ
[x,y,z]=sphere;            %����������N+1������N+1���ľ���ʹ��surf(X,Y,Z)����һ�����壬ȱʡʱNȡ20
set(0,'defaultFigurePosition',[100,100,1000,400]);    %�޸�ͼ��ͼ��λ�õ�Ĭ������
set(0,'defaultFigureColor', [1 1 1])                  %�޸�ͼ�α�����ɫ������
figure,
subplot(121),warp(I);              %��ʾͼ��ӳ�䵽����ƽ��
subplot(122),warp(x,y,z,I);              %����άͼ������ӳ����ά�������
grid;                                   %��������

