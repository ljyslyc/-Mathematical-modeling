close all;							%�رյ�ǰ����ͼ�δ��ڣ���չ����ռ��������������ռ����б���
clear all;
clc;
I=imread('hua.jpg');                         %IΪ���Ĳ�ɫͼ���������󻨵�ͼ���RGB������ֵ
R=I(:,:,1);                                  %��ɫ����
G=I(:,:,2);                                  %��ɫ���� 
B=I(:,:,3);                                   %��ɫ���� 
R=double(R);  G=double(G); B=double(B);     %����double()��������������תΪdouble��
Ravg1=mean2(R);                           %��ɫ������ֵ
Gavg1=mean2(G);                           %��ɫ������ֵ
Bavg1=mean2(B);                            %��ɫ������ֵ 
Rstd1=std(std(R));			                %��ɫ�����ķ���
Gstd1= std(std(G));		             	       %��ɫ�����ķ���
Bstd1=std(std(B));			                 %��ɫ�����ķ���
J=imread('yezi.jpg');                           %JΪҶ�ӵĲ�ɫͼ����������Ҷ�ӵ�ͼ���RGB������ֵ
R=J(:,:,1);                                    %��ɫ����
G=J(:,:,2);                                    %��ɫ���� 
B=J(:,:,3);                                     %��ɫ���� 
R=double(R);  G=double(G); B=double(B);       %����double()��������������תΪdouble��
Ravg2=mean2(R);                             %��ɫ������ֵ
Gavg2=mean2(G);                             %��ɫ������ֵ
Bavg2=mean2(B);                              %��ɫ������ֵ 
Rstd2=std(std(R));			                  %��ɫ�����ķ���
Gstd2= std(std(G));			                  %��ɫ�����ķ���
Bstd2=std(std(B));			                  %��ɫ�����ķ���
set(0,'defaultFigurePosition',[100,100,1000,500]);  %�޸�ͼ��ͼ��λ�õ�Ĭ������
set(0,'defaultFigureColor',[1 1 1])       
K=imread('flower1.jpg');figure;subplot(131),imshow(K); %��ʾԭͼ��  
subplot(132),imshow(I);                         %��ʾ����ͼ��  
subplot(133),imshow(J);                         %��ʾҶ�ӵ�ͼ��
 

