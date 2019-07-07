close all;                  %�رյ�ǰ����ͼ�δ��ڣ���չ����ռ��������������ռ����б���
clear all;
clc;
[X,map]=imread('trees.tif');                %����ͼ��
J1=imresize(X, 0.25);                        %�������ű�����ʵ������ͼ����ʾ
J2=imresize(X, 3.5);
J3=imresize(X, [64 40]);                   %�������ź�ͼ�����У�ʵ������ͼ����ʾ 
J4=imresize(X, [64 NaN]);
J5=imresize(X, 1.6, 'bilinear');          %����ͼ���ֵ������ʵ������ͼ����ʾ   
J6=imresize(X, 1.6, 'triangle');
[J7, newmap]=imresize(X,'Antialiasing',true,'Method','nearest',...
                      'Colormap','original','Scale', 0.15);%����ͼ����������ʵ������ͼ����ʾ
set(0,'defaultFigurePosition',[100,100,1000,500]);%�޸�ͼ��ͼ��λ�õ�Ĭ������
set(0,'defaultFigureColor',[1 1 1])%�޸�ͼ�α�����ɫ������
figure(1)                                         %��ʾ��������Ч��ͼ
subplot(121),imshow(J1);
subplot(122),imshow(J2);
figure(2)
subplot(121),imshow(J3);
subplot(122),imshow(J4);
figure(3)
subplot(121),imshow(J5);
subplot(122),imshow(J6);
figure(4),
subplot(121),imshow(X); 
subplot(122),imshow(J7);
