close all;                  %�رյ�ǰ����ͼ�δ��ڣ���չ����ռ��������������ռ����б���
clear all;
clc;
load wbarb;                 %����ͼ������
[C,S] = wavedec2(X,3,'db4');%����С���ֽ�
[thr,sorh,keepapp] = ddencmp('cmp','wv',X)%����ͼ��ѹ������Ҫ��һЩ����
[Xcomp,CXC,LXC,PERF0,PERFL2] =wdencmp('gbl',C,S,'db4',3,thr,sorh,keepapp);%���ղ���ѹ��ͼ�񣬲����ؽ��
disp('С��ϵ������0��ϵ�������ٷֱȣ�') %����ѹ������
PERFL2
disp('ѹ����ͼ��ʣ�������ٷֱȣ�')
PERF0
set(0,'defaultFigurePosition',[100,100,1000,500]);%�޸�ͼ��ͼ��λ�õ�Ĭ������
set(0,'defaultFigureColor',[1 1 1])       %�޸�ͼ�α�����ɫ������
figure;            %����ͼ��               
colormap(map);
subplot(121); image(X); axis square%��ʾѹ�����
subplot(122); image(Xcomp); axis square


