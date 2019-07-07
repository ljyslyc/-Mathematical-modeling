close all;                  %�رյ�ǰ����ͼ�δ��ڣ���չ����ռ��������������ռ����б���
clear all;
clc;
I = imread('peppers.png'); %����ͼ��
fun = @(block_struct) imrotate(block_struct.data,30);%��ȡ���������ĺ������
I1 = blockproc(I,[64 64],fun);              %���з�������
fun = @(block_struct) std2(block_struct.data) ;  %��ȡ��ȡ���������ĺ������
I2 = blockproc(I,[32 32],fun);%���з�������
fun = @(block_struct) block_struct.data(:,:,[3 1 2]);%��ȡ���������ĺ������
blockproc(I,[100 100],fun,'Destination','brg_peppers.tif');%���з�������
set(0,'defaultFigurePosition',[100,100,1000,500]);%�޸�ͼ��ͼ��λ�õ�Ĭ������
set(0,'defaultFigureColor',[1 1 1])%�޸�ͼ�α�����ɫ������
figure%��ʾ�������
subplot(131),imshow(I1);
subplot(132),imshow(I2,[]);
subplot(133),imshow('brg_peppers.tif');