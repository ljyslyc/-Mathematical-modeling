clear all;  %��������ռ䣬�ر�ͼ�δ��ڣ����������
close all;
clc;
I=imread('����1.jpg');              %ͼ������
I1=rgb2gray(I);                 %ת���ɻҶ�ͼ��I1        
I2=wiener2(I1,[5,5]);            %��ͼ�����ά���˲�I2  
I3=edge(I2,'sobel', 'horizontal');%��Sobelˮƽ���Ӷ�ͼ���Ե��
theta=0:179;    %����ѡ��Ƕ�
r=radon(I3,theta);%��ͼ�����Radon�任
[m,n]=size(r);
c=1;
for i=1:m
    for j=1:n
        if  r(1,1)<r(i,j)
           r(1,1)=r(i,j);
            c=j;
        end
    end
end                              %���Radon�任�����еķ�ֵ����Ӧ��������
rot=90-c;%ȷ����ת�Ƕ�
I4=imrotate(I2,rot,'crop');        %��ͼ�������ת����
set(0,'defaultFigurePosition',[100,100,1200,450]); %�޸�ͼ��ͼ��λ�õ�Ĭ������
set(0,'defaultFigureColor',[1 1 1])                 %�޸�ͼ�α�����ɫ������
figure,                                             %��ʾ������
subplot(121),imshow(I)
subplot(122),imshow(I2)
figure,
subplot(121),imshow(I3)
subplot(122),imshow(I4)