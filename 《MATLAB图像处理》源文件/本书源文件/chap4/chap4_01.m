close all;                            %�رյ�ǰ����ͼ�δ��ڣ���չ����ռ��������������ռ����б���
clear all;
clc;
gamma=0.5;                            %�趨�������Զ�ȡֵ
I=imread('peppers.png');			  %����Ҫ�����ͼ�񣬲���ֵ��I
R=I;                                  %��ͼ�����ݸ�ֵ��R
R (:,:,2)=0;                          %��ԭͼ���ɵ�ɫͼ�񣬱�����ɫ
R(:,:,3)=0;
R1=imadjust(R,[0.5 0.8],[0 1],gamma); %���ú���imadjust����R�ĻҶȣ��������R1
G=I;								  %��ͼ�����ݸ�ֵ��G
G(:,:,1)=0;							  %��ԭͼ���ɵ�ɫͼ�񣬱�����ɫ
G(:,:,3)=0;
G1=imadjust(G,[0 0.3],[0 1],gamma);	  %���ú���imadjust����G�ĻҶȣ��������G1
B=I;								  %��ͼ�����ݸ�ֵ��B
B(:,:,1)=0;							  %��ԭͼ���ɵ�ɫͼ�񣬱�����ɫ
B(:,:,2)=0;
B1=imadjust(B,[0 0.3],[0 1],gamma);	  %���ú���imadjust����B�ĻҶȣ��������B1
I1=R1+G1+B1;                          %��任���RGBͼ��  
set(0,'defaultFigurePosition',[100,100,1000,500]);%�޸�ͼ��ͼ��λ�õ�Ĭ������
set(0,'defaultFigureColor',[1 1 1])%�޸�ͼ�α�����ɫ������
figure(1)
subplot(121),imshow(R);%����R��R1��G��G1��B��B1ͼ�񣬹۲����ԻҶȱ任���
subplot(122),imshow(R1); 
figure(2);
subplot(121),imshow(G);
subplot(122),imshow(G1);
figure(3);
subplot(121),imshow(B);
subplot(122),imshow(B1);
figure(4);
subplot(121),imshow(I);
subplot(122),imshow(I1);