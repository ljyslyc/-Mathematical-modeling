clear all  %�����ǰ����ڣ��رյ�ǰ���ڣ��������
close all
clc
load woman; %����ͼ��
X1=X;map1=map;%����ͼ�����ݺ�ӳ��
load wbarb;%����ͼ��
X2=X;map2=map;%����ͼ�����ݺ�ӳ��
[C1,S1]=wavedec2(X1,2,'sym4');%ͼ���С���ֽ�
[C2,S2]=wavedec2(X2,2,'sym4');
C=1.2*C1+0.5*C2;             %��ͼ���С���ֽ��������ںϷ���1
C=0.4*C;
C0=0.2*C1+1.5*C2;   %��ͼ���С���ֽ��������ںϷ���2
C0=0.5*C;
xx1=waverec2(C,S1,'sym4');%��С���ֽ�Ľ�������ںϴ���
xx2=waverec2(C0,S2,'sym4');
set(0,'defaultFigurePosition',[100,100,1000,500]);%�޸�ͼ��ͼ��λ�õ�Ĭ������
set(0,'defaultFigureColor',[1 1 1])       %�޸�ͼ�α�����ɫ������
figure                 %����ͼ����ʾ����
colormap(map2),
subplot(121),image(X1) %��ʾԭͼ����ںϽ��
subplot(122),image(X2)
figure                 %����ͼ����ʾ����
colormap(map2),
subplot(121),image(xx1);
subplot(122),image(xx2);