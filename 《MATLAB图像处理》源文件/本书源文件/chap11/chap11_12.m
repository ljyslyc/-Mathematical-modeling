%����11-14��
I=imread('leaf1.bmp');                %����ͼ�����ݸ�ֵ��I
I=rgb2gray(I);                        %����ɫͼ���Ϊ�Ҷ�ͼ��
bwI=im2bw(I,graythresh(I));            %��ͼ����ж�ֵ������õ���ֵ��ͼ��ֵ��bwI
bwIsl=~bwI;                          %�Զ�ֵͼ��ȡ��
h=fspecial('average');                  %ѡ����ֵ�˲�        
bwIfilt=imfilter(bwIsl,h);                 %��ͼ�������ֵ�˲�
bwIfiltfh=imfill(bwIfilt,'holes');            %����ֵͼ��Ŀն�����
bdI=boundaries(bwIfiltfh,4,'cw');          %׷��4����Ŀ��߽�
d=cellfun('length',bdI);                   %��bdI��ÿһ��Ŀ��߽�ĳ��ȣ�����ֵd��һ������
[dmax,k]=max(d);                       %��������d������ֵ������max_d�У�kΪ������
B4=bdI{k(1)};                           %�����߽粻ֹһ������ȡ�����е�һ�����ɡ�B4��һ����������
[m,n]=size(bwIfiltfh);                     %���ֵͼ��Ĵ�С
xmin=min(B4(:,1));                       
ymin=min(B4(:,2));    
%����һ����ֵͼ��,��СΪm n��xmin,ymin��B4����С��x��y������                   
bim=bound2im(B4,m,n,xmin,ymin);         
[x,y]=minperpoly(bwIfiltfh,2);               %ʹ�ô�СΪ2�ķ��ε�Ԫ
b2=connectpoly(x,y);                     %��������(X,Y)˳ʱ�������ʱ�����ӳɶ����
B2=bound2im(b2,m,n,xmin,ymin);                     
set(0,'defaultFigurePosition',[100,100,1000,500]);%�޸�ͼ��ͼ��λ�õ�Ĭ������
set(0,'defaultFigureColor',[1 1 1])      
figure,subplot(121);imshow(bim);            %��ʾԭͼ��߽�
subplot(122),imshow(B2);                  %��ʾ����СΪ2�������ε�Ԫ���Ƶı߽�
