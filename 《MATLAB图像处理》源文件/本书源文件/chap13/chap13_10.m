close all;                  %�رյ�ǰ����ͼ�δ��ڣ���չ����ռ��������������ռ����б���
clear all;
clc;
load gatlin2;%װ�ز���ʾԭʼͼ��
init=2055615866;%���ɺ���ͼ����ʾ
randn('seed',init)
XX=X+2*randn(size(X));
[c,l]=wavedec2(XX,2,'sym4');%��ͼ��������봦��,��sym4С��������x��������ֽ�
a2=wrcoef2('a',c,l,'sym4',2); %�ع��ڶ���ͼ��Ľ���ϵ��
n=[1,2];%���ó߶�����
p=[10.28,24.08];%������ֵ����
nc=wthcoef2('t',c,l,n,p,'s');%�Ը�ƵС��ϵ��������ֵ����
mc=wthcoef2('t',nc,l,n,p,'s');%�ٴζԸ�ƵС��ϵ��������ֵ����
X2=waverec2(mc,l,'sym4');%%ͼ��Ķ�άС���ع�
set(0,'defaultFigurePosition',[100,100,1000,500]);%�޸�ͼ��ͼ��λ�õ�Ĭ������
set(0,'defaultFigureColor',[1 1 1])       %�޸�ͼ�α�����ɫ������
figure                                % ��ʾԭͼ�񼰴����Ժ���
colormap(map)
subplot(131),image(XX),axis square; 
subplot(132),image(a2),axis square;
subplot(133),image(X2),axis square;        
Ps=sum(sum((X-mean(mean(X))).^2));%���������
Pn=sum(sum((a2-X).^2));
disp('����С��2��ֽ�ȥ��������')
snr1=10*log10(Ps/Pn)   
disp('����С����ֵȥ��������')
Pn1=sum(sum((X2-X).^2));
snr2=10*log10(Ps/Pn1)