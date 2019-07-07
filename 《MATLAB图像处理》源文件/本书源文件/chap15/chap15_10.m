clear all;  						%��������ռ䣬�ر�ͼ�δ��ڣ����������
close all;
clc;
I=imread('girl1.bmp');
I1=refine_face_detection(I); 			%�����ָ�
I1=double(I1);
[m,n]=size(I1);
theta1=0;							%����
theta2=pi/2;
f = 0.88;							%����Ƶ��
sigma = 2.6;						%����
Sx = 5;
Sy = 5;							%����Ⱥͳ���
Gabor1=Gabor_hy(Sx,Sy,f,theta1,sigma);%����Gabor�任�Ĵ��ں���
Gabor2=Gabor_hy(Sx,Sy,f,theta2,sigma);%����Gabor�任�Ĵ��ں���
Regabout1=conv2(I1,double(real(Gabor1)),'same');
Regabout2=conv2(I1,double(real(Gabor2)),'same');
Regabout=(Regabout1+Regabout2)/2;
%% ��һ������
J1 = im2bw(Regabout,0.2);
SE1 = strel('square',2);BW = imdilate(J1,SE1);
[B,L,N] = bwboundaries(BW,'noholes');	%�߽����
a = zeros(1,N);
for i1 = 1:N
    a(i1) = length(find(L == i1));
end
a1 = find(a > 300);
for i1 = 1:size(a1,2)
L(find(L == a1(i1))) = 0;
end
L1 = double(uint8(L*255))/255;
a = 0;
BW = I1 .* L1;
%% �ڶ�������
for i2 = 1:m
    for j2 = 1:n
        if BW(i2,j2) > 0 && BW(i2,j2) < 50
            BW(i2,j2) = 255;
        end
    end
end
BW = uint8(BW);
J2 = im2bw(BW,0.8);
SE1 = strel('rectangle',[2 5]);BW = imdilate(J2,SE1);
[B,L,N] = bwboundaries(BW,'noholes');	%�߽����
a = zeros(1,N);
for i1 = 1:N
    a(i1) = length(find(L == i1));
end
a1 = find(a > 300);
for i1 = 1:size(a1,2)
L(find(L == a1(i1))) = 0;
end
L1 = double(uint8(L*255))/255;
a =0;
SE1 = strel('rectangle',[10 10]);BW = imdilate(L1,SE1);
BW = uint8(I1 .* double(BW));
set(0,'defaultFigurePosition',[100,100,1200,450]); %�޸�ͼ��ͼ��λ�õ�Ĭ������
set(0,'defaultFigureColor',[1 1 1])                %�޸�ͼ�α�����ɫ������
figure,
imshow(BW);
