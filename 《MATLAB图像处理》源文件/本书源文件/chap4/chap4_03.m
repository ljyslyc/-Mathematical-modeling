close all;                                  %�رյ�ǰ����ͼ�δ��ڣ���չ����ռ��������������ռ����б���
clear all;
clc
R=imread('peppers.png');                    %����ͼ�񣬸�ֵ��R
G=rgb2gray(R);                              %ת�ɻҶ�ͼ��
J=double(G);                                %��������ת����˫����
H=(log(J+1))/10;                             %���л��ڳ��ö����ķ����ԻҶȱ任
set(0,'defaultFigurePosition',[100,100,1000,500]);%�޸�ͼ��ͼ��λ�õ�Ĭ������
set(0,'defaultFigureColor',[1 1 1])%�޸�ͼ�α�����ɫ������
subplot(121),imshow(G);%��ʾͼ��
subplot(122),imshow(H);