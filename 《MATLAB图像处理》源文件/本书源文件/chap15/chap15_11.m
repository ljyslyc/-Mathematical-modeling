clear all;
close all;
clc
X1=[1 1 1 1, 1 0 0 1, 1 1 1 1, 1 0 0 1];  %ʶ��ģʽ
X2=[0 1 0 0, 0 1 0 0, 0 1 0 0, 0 1 0 0];
X3=[1 1 1 1, 1 0 0 1, 1 0 0 1, 1 1 1 1];
X=[X1;X2;X3];
Y1=[1 0 0];                           %���ģʽ           
Y2=[0 1 0];
Y3=[0 0 1];
Yo=[Y1;Y2;Y3];
n=16; %�������Ԫ����
p=8;  %�м����Ԫ����
q=3;  %�����Ԫ����
k=3 ;%ѵ��ģʽ����
a1=0.2; b1=0.2; %ѧϰϵ����
%rou=0.5;%����ϵ����
emax=0.01; cntmax=100;%�����ѵ������
[w,v,theta,r,t,mse]=bptrain(n,p,q,X,Yo,k,emax,cntmax,a1,b1);%���ú���bptrainѵ������
X4=[1 1 1 1, 1 0 0 1, 1 1 1 1, 1 0 1 1 ];
disp('ģʽX1��ʶ������')%���Բ���ʾ��ͼ�ε�ʶ����
c1=bptest(p,q,n,w,v,theta,r,X1)
disp('ģʽX2��ʶ������')
c2=bptest(p,q,n,w,v,theta,r,X2)
disp('ģʽX3��ʶ������')
c3=bptest(p,q,n,w,v,theta,r,X3)
disp('ģʽX4��ʶ������')
c4=bptest(p,q,n,w,v,theta,r,X4)
c=[c1;c2;c3;c4];
for i=1:4
    for j=1:3
       if c(i,j)>0.5
          c(i,j)=1;
      elseif c(i,j)<0.2
       c(i,j)=0;
       end
    end
end
disp('ģʽX1~X4��ʶ������')
c