close all;              %�رյ�ǰ����ͼ�δ��ڣ���չ����ռ��������������ռ����б���
clear all;
clc;
R=imread('peppers.png');%����ԭͼ�񣬸�ֵ��R
J=rgb2gray(R);          %����ɫͼ������Rת��Ϊ�Ҷ�ͼ������J
[M,N]=size(J);          %��ûҶ�ͼ������J��������M��N
x=1;y=1;                %��������������x������������y    
for x=1:M
    for y=1:N
        if (J(x,y)<=35);     %�ԻҶ�ͼ��J���зֶδ��������Ľ�����ظ�����H
            H(x,y)=J(x,y)*10;
        elseif(J(x,y)>35&J(x,y)<=75);
            H(x,y)=(10/7)*[J(x,y)-5]+50;
        else(J(x,y)>75);
            H(x,y)=(105/180)*[J(x,y)-75]+150;
        end
    end
end
set(0,'defaultFigurePosition',[100,100,1000,500]);%�޸�ͼ��ͼ��λ�õ�Ĭ������
set(0,'defaultFigureColor',[1 1 1])%�޸�ͼ�α�����ɫ������
subplot(121),imshow(J)%��ʾ����ǰ���ͼ��
subplot(122),imshow(H);