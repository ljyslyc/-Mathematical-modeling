close all;                  %�رյ�ǰ����ͼ�δ��ڣ���չ����ռ��������������ռ����б���
clear all;
clc;
load wmandril;            %����ͼ������
nbc=size(map,1);          %��ȡ��ɫӳ�����       
Y=wcodemat(X,nbc);%��ͼ�����ֵ�������α��ɫ����
[C,S]=wavedec2(X,2,'db4'); %��ͼ��С���ֽ�
thr=20;                  %������ֵ
[Xcompress1,cxd,lxd,perf0,perfl2]=wdencmp('gbl',C,S,'db4',2,thr,'h',1);%��ͼ�����ȫ��ѹ��
Y1=wcodemat(Xcompress1,nbc); %��ͼ�����ݽ���α��ɫ����
set(0,'defaultFigurePosition',[100,100,1000,500]);%�޸�ͼ��ͼ��λ�õ�Ĭ������
set(0,'defaultFigureColor',[1 1 1])       %�޸�ͼ�α�����ɫ������
figure                 %����ͼ����ʾ����
colormap(gray(nbc));       %����ӳ����ͼ�ȼ�
subplot(121),image(Y),axis square %��ʾ
subplot(122);image(Y1),axis square
disp('С��ϵ������0��ϵ�������ٷֱȣ�')   %���ѹ�����ʱ���
perfl2
disp('ѹ����ͼ��ʣ�������ٷֱȣ�')
perf0
