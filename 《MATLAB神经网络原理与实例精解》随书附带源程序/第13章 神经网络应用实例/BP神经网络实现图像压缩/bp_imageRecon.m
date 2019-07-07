% bp_imageRecon.m

%% ����
clear,clc
close all

%% ��������
col=256;
row=256;
I=imread('d:\lena.bmp');
load comp
com.lw=double(com.lw)/63;
com.b=double(com.b)/63;
com.d=double(com.d)/63;
com.lw=com.lw*(maxlw-minlw)+minlw;
com.b=com.b*(maxb-minb)+minb;
com.d=com.d*(maxd-mind)+mind;

%% �ؽ�
for i=1:4096
   Y(:,i)=com.lw*(com.d(:,i)) +com.b;
end

%% ����һ��
Y=uint8(Y*255);

%% ͼ���ָ�
I1=re_divide(Y,col,4);

%% ��������
fprintf('PSNR :\n  ');
psnr=10*log10(255^2*row*col/sum(sum((I-I1).^2)));
disp(psnr)
a=dir();
for i=1:length(a)
   if (strcmp(a(i).name,'comp.mat')==1) 
       si=a(i).bytes;
       break;
   end
end
fprintf('rate: \n  ');
rate=double(si)/(256*256);
disp(rate)
figure(1)
imshow(I)
title('ԭʼͼ��');
figure(2)
imshow(I1)
title('�ؽ�ͼ��'); 
