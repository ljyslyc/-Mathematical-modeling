function [w,v]=bptrain(n,p,q,X,Y,k,emax,cntmax)
clear all;
close all;
clc
tic
n='������Ԫ����:16';   %�������Ԫ
disp(n)
p='�м����Ԫ����:8';    %�м����Ԫ 
disp(p)
q='�������Ԫ����:3';    %�������Ԫ
disp(q)
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
w=rands(n,p);%�����������������Ȩ
v=rands(p,q); %�����������������Ȩ
theta=rands(1,p);%�м�����ֵ
r=rands(1,q);%��������ֵ
a1=0.1, b1=0.1, %ѧϰϵ����
E=zeros(1,q);
emax=0.01, cntmax=100;%�����ѵ������
cnt=1;
er=0;%ȫ�����Ϊ��
while ((er>emax)|(cnt<=cntmax))
 %ѭ��ʶ��ģʽ 
 for cp=1:k
     X0=X(cp,:);             
     Y0=Yo(cp,:);
     %�����м�������Y(j) 
     Y=X0*w; 
     %�����м������b
     Y=Y-theta;    %�м����ֵ
     for j=1:p
         b(j)=1/(1+exp(-Y(j)));%�м�����f(sj)
     end      
    %������������c
             Y=b*v;
             Y=Y-r;  % �������ֵ
        for t=1:3
           c(t)=1/(1+exp(-Y(t))); %��������
        end 
    %���������У�����d
        for t=1:3 
          d(t)=(Y0(t)-c(t))*c(t)*(1-c(t));
        end
   %�����м��У�����e
         xy=d*v';
         for t=1:8
           e(t)=xy(t)*b(t)*(1-b(t));
         end
   %������һ�ε��м��������֮���µ�����Ȩv(i,j),��ֵt2(j)
          for t=1:q
              for j=1:p
                  v(j,t)=v(j,t)+a1*d(t)*b(j);
              end
              r(t)=r(t)+a1*d(t);
          end
      %������һ�ε��������м��֮���µ�����Ȩw(i,j),��ֵt1(j)
           for j=1:p
              for i=1:n
                  w(i,j)=w(i,j)+b1*e(j)*X0(i)
              end
              theta(j)=theta(j)+b1*e(j)
           end
           for t=1:q
               E(cp)=(Y0(t)-c(t))*(Y0(t)-c(t))+E(cp);%��ǰѧϰģʽ��ȫ�����
           end
            E(cp)=E(cp)*0.5;
    %������һģʽ    
 end
  er=max(E);%����ȫ�����
  cnt=cnt+1;%����ѧϰ����
end
 time=toc;
 %����ѵ�� 
 %S='ѵ������:';
 %disp(S);
 %disp(cnt);
 %S='�����Ȩֵ:';
 %disp(S);
 %disp(v);
 %S='�м��Ȩֵ:';
 %disp(S);
 %disp(w);