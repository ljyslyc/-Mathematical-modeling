close all;                  %�رյ�ǰ����ͼ�δ��ڣ���չ����ռ��������������ռ����б���
clear all;     %�رյ�ǰ����ͼ�δ��ڣ���չ����ռ��������������ռ����б���
clc;
load detfingr; %����ͼ������
nbc=size(map,1);%��ȡ��ɫӳ�����
[c,s]=wavedec2(X,3,'sym4');%��ͼ������X����3��С���ֽ� ����С������sym4
alpha=1.5;%���ò���alpha��m������wdcbm2����ͼ��ѹ���ķֲ���ֵ
m=2.7*prod(s(1,:)); 
[thr,nkeep]=wdcbm2(c,s,alpha,m)
[xd,cxd,sxd,perf0,perfl2] =wdencmp('lvd',c,s,'sym4',3,thr,'h');
set(0,'defaultFigurePosition',[100,100,1000,500]);%�޸�ͼ��ͼ��λ�õ�Ĭ������
set(0,'defaultFigureColor',[1 1 1])       %�޸�ͼ�α�����ɫ������
figure                 %����ͼ����ʾ����
colormap(pink(nbc));
subplot(121), image(wcodemat(X,nbc)),
subplot(122), image(wcodemat(xd,nbc)),
disp('С��ϵ������0��ϵ�������ٷֱȣ�')   %���ѹ�����ʱ���
perfl2
disp('ѹ����ͼ��ʣ�������ٷֱȣ�')
perf0
