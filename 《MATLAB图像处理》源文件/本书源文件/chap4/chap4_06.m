close all;                          %�رյ�ǰ����ͼ�δ��ڣ���չ����ռ��������������ռ����б���
clear all;
clc;
RGB=imread('eight.tif');            %����eightͼ�񣬸�ֵ��RGB
A=imnoise(RGB,'gaussian',0,0.05);    %�����˹������
I=A;                                %��A��ֵ��I
M=3;                                %���õ����Ӵ���M
I=im2double(I);                     %��I��������ת����˫����
RGB=im2double(RGB);
for i=1:M
   I=imadd(I,RGB);                  %����ԭͼ���������ͼ����ж�ε��ӣ�������ظ�I
end
avg_A=I/(M+1);                      %����ӵ�ƽ��ͼ�� 
set(0,'defaultFigurePosition',[100,100,1000,500]);%�޸�ͼ��ͼ��λ�õ�Ĭ������
set(0,'defaultFigureColor',[1 1 1])%�޸�ͼ�α�����ɫ������
subplot(121); imshow(A);   %��ʾ���뽷���������ͼ��
subplot(122); imshow(avg_A);    %��ʾ��������������ͼ��
