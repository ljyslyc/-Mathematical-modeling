close all; clear all; clc;						%�ر�����ͼ�δ��ڣ���������ռ����б��������������
RGB = imread('board.tif');						%����RGBͼ��
YCBCR = rgb2ycbcr(RGB);					%��RGBͼ��ת��ΪYCBCRͼ��
figure;
subplot(121), imshow(RGB)					%��ʾRGBͼ��
subplot(122), imshow(YCBCR)					%��ʾYCBCRͼ��
