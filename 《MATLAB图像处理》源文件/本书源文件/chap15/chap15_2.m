close all;                  %�رյ�ǰ����ͼ�δ��ڣ���չ����ռ��������������ռ����б���
clear all;
clc;
proj1=90,N1=128;%����ͶӰ���ݴ�С
degree1=projdata(proj1,N1);%���ú���projdata����ͷģ�͵�ͶӰ����
proj2=180,N2=256;%����ͶӰ���ݴ�С
degree2=projdata(proj2,N2);%���ú���projdata����ͷģ�͵�ͶӰ����
set(0,'defaultFigurePosition',[100,100,1200,450]); %�޸�ͼ��ͼ��λ�õ�Ĭ������
set(0,'defaultFigureColor',[1 1 1])                 %�޸�ͼ�α�����ɫ������
figure, 
subplot(121),pcolor(degree1)%��ʾ180*128ͷģ��    
subplot(122),pcolor(degree2)%��ʾ180*256ͷģ��    


