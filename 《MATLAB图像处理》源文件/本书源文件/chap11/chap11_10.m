%����11-10��
close all;							%�رյ�ǰ����ͼ�δ��ڣ���չ����ռ��������������ռ����б���
clear all;
clc;
I = imread('wall.jpg');                        %��ȡͼ�񣬲���ֵ��I
I=rgb2gray(I);                              %��ɫͼ���Ϊ�Ҷ�ͼ��
[G,gabout] = gaborfilter(I,2,4,16,pi/10);        %����garborfilter()������ͼ����С���任
J=fft2(gabout);                             %���˲����ͼ����FFT�任���任��Ƶ�� 
A=double(J);
[m,n]=size(A);
B=A;
C=zeros(m,n);
for i=1:m-1
    for j=1:n-1
        B(i,j)=A(i+1,j+1);
        C(i,j)=abs(round(A(i,j)-B(i,j)));
    end
end
h=imhist(mat2gray(C))/(m*n);                  %�Ծ���C��һ�����������Ҷ�ֱ��ͼ���õ���һ����ֱ��ͼ
mean=0;con=0;ent=0;
for i=1:256                                  %��ͼ��ľ�ֵ���ԱȶȺ���
    mean=mean+(i*h(i))/256;
    con=con+i*i*h(i);
    if(h(i)>0)
        ent=ent-h(i)*log2(h(i));
    end
end
set(0,'defaultFigurePosition',[100,100,1000,500]);%�޸�ͼ��ͼ��λ�õ�Ĭ������
set(0,'defaultFigureColor',[1 1 1])
figure;subplot(121);imshow(I);
subplot(122);imshow(uint8(gabout));
mean,con,ent
