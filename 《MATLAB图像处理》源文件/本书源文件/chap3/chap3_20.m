close all;                          %�رյ�ǰ����ͼ�δ���
clear all;                           %��չ����ռ����
clc;                               %����
RGB = imread('peppers.png');        %��ȡͼ����Ϣ
c = [12 146 410];                   %�½�һ������c���������������
r = [104 156 129];                   %�½�һ������r��������غ�����
set(0,'defaultFigurePosition',[100,100,1000,500]);    %�޸�ͼ��ͼ��λ�õ�Ĭ������
set(0,'defaultFigureColor', [1 1 1])                  %�޸�ͼ�α�����ɫ������
pixels1=impixel(RGB)               %����ʽ�����ѡ������
pixels2= impixel(RGB,c,r)            %������������Ϊ�����������ʾ�ض����ص���ɫֵ
