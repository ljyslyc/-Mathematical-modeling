close all; clear all; clc;						%�ر�����ͼ�δ��ڣ���������ռ����б��������������
RGB=reshape(ones(64,1)*reshape(jet(64),1,192),[64,64,3]);		%������ɫ���ߴ�Ϊ������
HSV=rgb2hsv(RGB);										%��RGBͼ��ת��ΪHSVͼ��
H=HSV(:,:,1);											%��ȡH����
S=HSV(:,:,2);											%��ȡS����
V=HSV(:,:,3);											%��ȡV����
figure(1)
subplot(121), imshow(H)									%��ʾHͼ��
subplot(122), imshow(S)									%��ʾSͼ��
figure(2)
subplot(121), imshow(V)									%��ʾVͼ��
subplot(122), imshow(RGB)								%��ʾRGBͼ��
