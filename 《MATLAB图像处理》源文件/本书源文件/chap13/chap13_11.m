close all;                  %�رյ�ǰ����ͼ�δ��ڣ���չ����ռ��������������ռ����б���
clear all;
clc;
load flujet;           %װ�ز���ʾԭʼͼ��
init=2055615866;        %���ɺ�����ͼ����ʾ
XX=X+8*randn(size(X));
n=[1,2];                %���ó߶�����
p=[10.28,24.08];        %������ֵ����
[c,l]=wavedec2(XX,2,'db2');%��С������db2��ͼ��XX����2��ֽ�
nc=wthcoef2('t',c,l,n,p,'s');%�Ը�ƵС��ϵ��������ֵ����
mc=wthcoef2('t',nc,l,n,p,'s');%�ٴζԸ�ƵС��ϵ��������ֵ����
X2=waverec2(mc,l,'db2');%%ͼ��Ķ�άС���ع�

[c1,l1]=wavedec2(XX,2,'sym4');%������С������sym4��ͼ��XX����2��ֽ�
nc1=wthcoef2('t',c1,l1,n,p,'s');%�Ը�ƵС��ϵ��������ֵ����
mc1=wthcoef2('t',nc1,l1,n,p,'s');%�ٴζԸ�ƵС��ϵ��������ֵ����
X3=waverec2(mc1,l1,'sym4');%%ͼ��Ķ�άС���ع�
set(0,'defaultFigurePosition',[100,100,1000,500]);%�޸�ͼ��ͼ��λ�õ�Ĭ������
set(0,'defaultFigureColor',[1 1 1])       %�޸�ͼ�α�����ɫ������
figure                                % ��ʾԭͼ�񼰴����Ժ���
colormap(map)
subplot(121);image(X);axis square;
subplot(122);image(XX);axis square; 
figure
colormap(map)
subplot(121);image(X2);axis square;
subplot(122);image(X3);axis square;
Ps=sum(sum((X-mean(mean(X))).^2));%���������
Pn=sum(sum((XX-X).^2));
Pn1=sum(sum((X2-X).^2));
Pn2=sum(sum((X3-X).^2));
disp('δ����ĺ�����ͼ�������')
snr=10*log10(Ps/Pn)
disp('����db2����С��ȥ���ͼ�������')
snr1=10*log10(Ps/Pn1)
disp('����sym4����С��ȥ���ͼ�������')
snr2=10*log10(Ps/Pn2)
