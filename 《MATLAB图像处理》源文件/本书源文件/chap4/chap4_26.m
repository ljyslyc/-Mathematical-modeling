close all;                  %�رյ�ǰ����ͼ�δ��ڣ���չ����ռ��������������ռ����б���
clear all;
clc;
[I,map]=imread('peppers.png');  %����ͼ��
Ta = maketform('affine', ...
[cosd(30) -sind(30) 0; sind(30) cosd(30) 0; 0 0 1]');% ������ת�����ṹ��
Ia = imtransform(I,Ta);                              %ʵ��ͼ����ת  
Tb = maketform('affine',[5 0 0; 0 10.5 0; 0 0 1]'); %�������Ų����ṹ��
Ib = imtransform(I,Tb);%ʵ��ͼ������  
xform = [1 0 55; 0 1 115; 0 0 1]';                    %����ͼ��ƽ�Ʋ����ṹ��
Tc = maketform('affine',xform);
Ic = imtransform(I,Tc, 'XData', ...                   %����ͼ��ƽ��
[1 (size(I,2)+xform(3,1))], 'YData', ...
[1 (size(I,1)+xform(3,2))],'FillValues', 255 );
Td = maketform('affine',[1 4 0; 2 1 0; 0 0 1]');% ����ͼ�������б�Ĳ����ṹ��
Id = imtransform(I,Td,'FillValues', 255);                          %ʵ��ͼ�������б�
set(0,'defaultFigurePosition',[100,100,1000,500]);%�޸�ͼ��ͼ��λ�õ�Ĭ������
set(0,'defaultFigureColor',[1 1 1])%�޸�ͼ�α�����ɫ������
figure                 %��ʾ���
subplot(121),imshow(Ia),axis on;
subplot(122),imshow(Ib),axis on;
figure
subplot(121),imshow(Ic),axis on;
subplot(122),imshow(Id),axis on;
