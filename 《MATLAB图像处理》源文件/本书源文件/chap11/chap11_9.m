%����11-9��
close all;			               %�رյ�ǰ����ͼ�δ��ڣ���չ����ռ��������������ռ����б���
clear all;
clc;
I = imread('wall.jpg');               %����ͼ��
I=rgb2gray(I);                        %ͼ���Ϊ�Ҷ�ͼ��
wall=fft2(I);                         %��ͼ�������ٸ���Ҷ�任
s=fftshift(wall);                     %���任���ͼ��Ƶ�����ĴӾ����ԭ���Ƶ����������
s=abs(s);
[nc,nr]=size(s);
x0=floor(nc/2+1);
y0=floor(nr/2+1);
rmax=floor(min(nc,nr)/2-1);
srad=zeros(1,rmax);
srad(1)=s(x0,y0);
thetha=91:270;                           %thethaȡֵ91-270
for r=2:rmax                             %ѭ���������Ƶ������
    [x,y]=pol2cart(thetha,r);
    x=round(x)'+x0;
    y=round(y)'+y0;
    for j=1:length(x)
        srad(r)=sum(s(sub2ind(size(s),x,y)));
    end
end
[x,y]=pol2cart(thetha,rmax);
x=round(x)'+x0;
y=round(y)'+y0;
sang=zeros(1,length(x));
for th=1:length(x)
    vx=abs(x(th)-x0);
    vy=abs(y(th)-y0);
    if((vx==0)&(vy==0))
        xr=x0;
        yr=y0;
    else
        m=(y(th)-y0)/(x(th)-x0)
        xr=(x0:x(th)).'
        yr=round(y0+m*(xr-x0));
    end
    for j=1:length(xr)
        sang(th)=sum(s(sub2ind(size(s),xr,yr)));
    end
end
set(0,'defaultFigurePosition',[100,100,1000,500]);	%�޸�ͼ��ͼ��λ�õ�Ĭ������
set(0,'defaultFigureColor',[1 1 1])
figure;subplot(121);
imshow('wall.jpg');subplot(122);                     %��ʾԭͼ
imshow(log(abs(wall)),[]);                           %��ʾƵ��ͼ
figure;subplot(121);plot(srad);                      %��ʾ
subplot(122);plot(sang);                             %��ʾ


