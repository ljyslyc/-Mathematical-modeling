close all;                  %�رյ�ǰ����ͼ�δ��ڣ���չ����ռ��������������ռ����б���
clear all;
clc;
[X,map]=imread('trees.tif');                %����ͼ��
R=map(X+1,1);R=reshape(R,size(X));          %��ȡͼ��R��Ϣ
G=map(X+1,2);G=reshape(G,size(X));          %��ȡͼ��G��Ϣ
B=map(X+1,3);B=reshape(B,size(X));          %��ȡͼ��B��Ϣ
Xrgb=0.2990*R+0.5870*G+0.1140*B;            %��RGB��ϳɵ�ͨ��
n=64                                        %���ûҶȼ�
X1=round(Xrgb*(n-1))+1;                     %����ͨ����ɫ��Ϣ��ת����64�Ҷȼ�
map2=gray(n);
set(0,'defaultFigurePosition',[100,100,1000,500]);%�޸�ͼ��ͼ��λ�õ�Ĭ������
set(0,'defaultFigureColor',[1 1 1])       %�޸�ͼ�α�����ɫ������
figure(1),image(X1);
colormap(map2);