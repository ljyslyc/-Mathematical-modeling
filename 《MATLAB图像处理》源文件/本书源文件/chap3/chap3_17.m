close all;                                    %�رյ�ǰ����ͼ�δ���
clear all;                                     %��չ����ռ����
clc;                                         %����
I=zeros(128,128,1,27);                        %������ά����I
for i=1:27                                    
[I(:,:,:,i),map]=imread('mri.tif',i);              %��ȡ��֡ͼ�����У����������I��
end
set(0,'defaultFigurePosition',[100,100,1000,500]);    %�޸�ͼ��ͼ��λ�õ�Ĭ������
set(0,'defaultFigureColor', [1 1 1])                  %�޸�ͼ�α�����ɫ������
montage(I,map);                             %����֡ͼ��ͬʱ��ʾ
