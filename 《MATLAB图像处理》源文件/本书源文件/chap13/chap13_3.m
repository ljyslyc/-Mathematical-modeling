close all;                  %�رյ�ǰ����ͼ�δ��ڣ���չ����ռ��������������ռ����б���
clear all;
clc;
set(0,'defaultFigurePosition',[100,100,1000,500]);%�޸�ͼ��ͼ��λ�õ�Ĭ������
set(0,'defaultFigureColor',[1 1 1])  
iter = 4;            %���ò�������
wav1 = 'db4';        %����С��
wav2 = 'bior1.3'; 
[s,w1,w2,w3,xyval] = wavefun2(wav1,iter,'plot');%�����άС������ʾ
[s1,w11,w21,w31,xyval1] = wavefun2(wav2,iter,'plot');