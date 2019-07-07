close all;                  %�رյ�ǰ����ͼ�δ��ڣ���չ����ռ��������������ռ����б���
clear all;
clc;
X=imread('flower.tif');         %��ȡͼ����� �Ҷ�ת��
X=rgb2gray(X);
[c,s] = wavedec2(X,2,'db4');    %��ͼ�����С��2��ֽ�
siz = s(size(s,1),:);           %��ȡ��2��С���ֽ�ϵ�������С
ca2 = appcoef2(c,s,'db4',2);    %��ȡ��1��С���ֽ�Ľ���ϵ��
chd2 = detcoef2('h',c,s,2);     %��ȡ��1��С���ֽ��ϸ��ϵ��ˮƽ����
cvd2 = detcoef2('v',c,s,2);     %��ȡ��1��С���ֽ��ϸ��ϵ����ֱ����    
cdd2 = detcoef2('d',c,s,2);     %��ȡ��1��С���ֽ��ϸ��ϵ���ԽǷ���
a2 = upcoef2('a',ca2,'db4',2,siz); %���ú���upcoef2����ȡ2��С��ϵ�������ع�
hd2 = upcoef2('h',chd2,'db4',2,siz); 
vd2 = upcoef2('v',cvd2,'db4',2,siz);
dd2 = upcoef2('d',cdd2,'db4',2,siz);
A1=a2+hd2+vd2+dd2;
[ca1,ch1,cv1,cd1] = dwt2(X,'db4');    %��ͼ�����С������ֽ�
a1 = upcoef2('a',ca1,'db4',1,siz);   %���ú���upcoef2����ȡ1��С���ֽ�ϵ�������ع�
hd1 = upcoef2('h',cd1,'db4',1,siz); 
vd1 = upcoef2('v',cv1,'db4',1,siz);
dd1 = upcoef2('d',cd1,'db4',1,siz);
A0=a1+hd1+vd1+dd1;
set(0,'defaultFigurePosition',[100,100,1000,500]);%�޸�ͼ��ͼ��λ�õ�Ĭ������
set(0,'defaultFigureColor',[1 1 1])       %�޸�ͼ�α�����ɫ������
figure                                 %��ʾ����˲���
subplot(141);imshow(uint8(a2));
subplot(142);imshow(hd2);
subplot(143);imshow(vd2);
subplot(144);imshow(dd2);
figure
subplot(141);imshow(uint8(a1));
subplot(142);imshow(hd1);
subplot(143);imshow(vd1);
subplot(144);imshow(dd1);
figure
subplot(131);imshow(X);
subplot(132);imshow(uint8(A1));
subplot(133);imshow(uint8(A0));

