close all;                  %�رյ�ǰ����ͼ�δ��ڣ���չ����ռ��������������ռ����б���
clear all;     %�رյ�ǰ����ͼ�δ��ڣ���չ����ռ��������������ռ����б���
clc;
load detfingr; %����ͼ������
nbc=size(map,1);
[C,S]=wavedec2(X,2,'db4');%ͼ��С���ֽ�
thr_h=[21 46];            %����ˮƽ������ֵ
thr_d=[21 46];            %���öԽǷ�����ֵ
thr_v=[21 46];            %���ô�ֱ������ֵ   
thr=[thr_h;thr_d;thr_v]; 
[Xcompress2,cxd,lxd,perf0,perfl2]=wdencmp('lvd',X,'db3',2,thr,'h');%���зֲ�ѹ��
set(0,'defaultFigurePosition',[100,100,1000,500]);%�޸�ͼ��ͼ��λ�õ�Ĭ������
set(0,'defaultFigureColor',[1 1 1])       %�޸�ͼ�α�����ɫ������
Y=wcodemat(X,nbc);
Y1=wcodemat(Xcompress2,nbc);
figure %��ʾԭͼ���ѹ��ͼ��
colormap(map)
subplot(121),image(Y),axis square
subplot(122),image(Y1),axis square
figure
%colormap(map)
subplot(121),image(Y),axis square
subplot(122),image(Y1),axis square
disp('С��ϵ������0��ϵ�������ٷֱȣ�')  %��ʾѹ������
perfl2
disp('ѹ����ͼ��ʣ�������ٷֱȣ�')
perf0
