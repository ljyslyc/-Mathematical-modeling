close all;							%�رյ�ǰ����ͼ�δ��ڣ���չ����ռ��������������ռ����б���
clear all;
clc;
I = imread('cameraman.tif');     %��ȡͼ��,��ֵ��I
J = filter2(fspecial('prewitt'), I); %��ͼ�����I�����˲�
set(0,'defaultFigurePosition',[100,100,1000,500]);%�޸�ͼ��ͼ��λ�õ�Ĭ������
set(0,'defaultFigureColor',[1 1 1])%�޸�ͼ�α�����ɫ������
K = imabsdiff(double(I),J);     %���˲����ͼ����ԭͼ��ľ���ֵ��
figure,                          %��ʾͼ�񼰽��
subplot(131),imshow(I);       
subplot(132),imshow(J,[]);
subplot(133),imshow(K,[]);
X =[ 255 10 75; 44 225 100];%�������,���ݸ�ʽdouble
Y =[ 50 50 50; 50 50 50 ];
X1 =uint8([ 255 10 75; 44 225 100]);%����������ݸ�ʽuint8
Y1 =uint8([ 50 50 50; 50 50 50 ]);
Z=imabsdiff(X,Y)%�����ֵ�Ĳ�
Z1=abs(X-Y)     %���ú���abs�������ֵ��            
Z2=abs(X1-Y1)
disp('Z��Z1�ȽϽ����'),Z_Z1=(Z==Z1)%�Ƚϲ�ͬ��������������ָ������
disp('Z��Z2�ȽϽ����'),Z_Z2=(Z==Z2)