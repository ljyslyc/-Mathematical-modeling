close all;                        %�رյ�ǰ����ͼ�δ��ڣ���չ����ռ��������������ռ����б���
clear all;
clc;
load woman;                          %��ȡ������ͼ������.
nbcol = size(map,1);                 %��ȡ��ɫӳ��������
[cA1,cH1,cV1,cD1] = dwt2(X,'db1');   %��ͼ������X����db1С�������е���ͼ��ֽ�
sX = size(X);                        %��ȡԭͼ���С
A0 = idwt2(cA1,cH1,cV1,cD1,'db4',sX);% ��С���ֽ�ĵ�һ��ϵ�������ع�
set(0,'defaultFigurePosition',[100,100,1000,500]);%�޸�ͼ��ͼ��λ�õ�Ĭ������
set(0,'defaultFigureColor',[1 1 1])       %�޸�ͼ�α�����ɫ������
subplot(131),imshow(uint8(X));            %��ʾ������
subplot(132),imshow(uint8(A0));
subplot(133),imshow(uint8(X-A0));

