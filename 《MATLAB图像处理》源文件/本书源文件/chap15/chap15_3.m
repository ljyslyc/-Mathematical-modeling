clear all;  			%��������ռ䣬�ر�ͼ�δ��ڣ����������
close all;
clc;
N=64;				    %��������ֵN
m=15;
L=2.0;
[x,h]=RLfilter(N,L)
x1=x(N-m:N+m);
h1=h(N-m:N+m);
set(0,'defaultFigurePosition',[100,100,1200,450]);%�޸�ͼ��ͼ��λ�õ�Ĭ������
set(0,'defaultFigureColor',[1 1 1])       %�޸�ͼ�α�����ɫ������
figure,                                              
subplot(121),
plot(x,h),axis tight,grid on  %��ʾ����
subplot(122),
plot(x1,h1),axis tight,grid on %��ʾ����
