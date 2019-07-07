close all;                  %�رյ�ǰ����ͼ�δ��ڣ���չ����ռ��������������ռ����б���
clear all;
clc;
[phi,psi,xval]=wavefun('sym4',10);          %�õ�sym4�ĳ߶Ⱥ�����С������
set(0,'defaultFigurePosition',[100,100,1000,500]);%�޸�ͼ��ͼ��λ�õ�Ĭ������
set(0,'defaultFigureColor',[1 1 1])  
subplot(121);plot(xval,phi,'k');        %��ʾ�߶Ⱥ���
axis([0 8 -0.5 1.3]);
axis square;
subplot(122); plot(xval,psi,'k');        %��ʾС������
axis([0 8 -1.5 1.6]);
axis square; 
[lo_d,hi_d,lo_r,hi_r]=wfilters('sym4');         %�õ�sym4������˲���
figure                                 %��ʾ����˲���
subplot(121);stem(lo_d,'.k');
subplot(122);stem(hi_d,'.k');
figure
subplot(121);stem(lo_r,'.k');
subplot(122);stem(hi_r,'.k');
