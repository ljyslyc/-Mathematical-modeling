close all;%�رյ�ǰ����ͼ�δ��ڣ���չ����ռ��������������ռ����б���
clc;
clear all;
I=imread('girl.bmp');           %����ͼ�񣬸�ֵ��I��J
J=imread('lenna.bmp');
I1=im2bw(I);                    %ת��Ϊ��ֵͼ��
J1=im2bw(J);
H=~(I1|J1);
G=~(I1&J1);
set(0,'defaultFigurePosition',[100,100,1000,500]);%�޸�ͼ��ͼ��λ�õ�Ĭ������
set(0,'defaultFigureColor',[1 1 1])%�޸�ͼ�α�����ɫ������
figure(1),%��ʾԭͼ����Ӧ�Ķ�ֵͼ��
subplot(121),imshow(I1),axis on;
subplot(122),imshow(J1),axis on;
figure(2), %��ʾ�����Ժ��ͼ��
subplot(121),imshow(H),axis on;  
subplot(122),imshow(G),axis on;


