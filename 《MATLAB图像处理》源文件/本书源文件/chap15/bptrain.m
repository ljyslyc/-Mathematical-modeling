function [w,v,theta,r,t,mse]=bptrain(n,p,q,X,Yo,k,emax,cntmax,a1,b1)
%n��ʾ������Ԫ����,p��ʾ�м����Ԫ������q��ʾ�����Ԫ����,
%X��ʾ����ѵ��ģʽ��Yo��ʾ��׼�����k��ʾѵ��ģʽ�ĸ���
%emax��ʾ�����cntmax��ʾ���ѵ��������a1��b1��ʾѧϰϵ����rou��ʾ����ϵ��
%w��theta��ʾѵ����������������м������Ȩϵ������ֵ��
%v��r��ʾѵ���������м�������������Ȩ����ֵ
%t��ʾѵ��ʱ��
tic
w=rands(n,p);%�����������������Ȩ
v=rands(p,q); %�����������������Ȩ
theta=rands(1,p);%�м�����ֵ
r=rands(1,q);%��������ֵ
cnt=1;
mse=zeros(1,cntmax);%ȫ�����Ϊ��
er=0;
while ((er>emax)|(cnt<=cntmax))
 E=zeros(1,q);
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
        for t=1:q
           c(t)=1/(1+exp(-Y(t))); %��������
        end 
    %���������У�����d
        for t=1:q 
          d(t)=(Y0(t)-c(t))*c(t)*(1-c(t));
        end
   %�����м��У�����e
         xy=d*v';
         for t=1:p
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
                  w(i,j)=w(i,j)+b1*e(j)*X0(i);
              end
              theta(j)=theta(j)+b1*e(j);
           end
           for t=1:q
               E(cp)=(Y0(t)-c(t))*(Y0(t)-c(t))+E(cp);%��ǰѧϰģʽ��ȫ�����
           end
           E(cp)=E(cp)*0.5;
    %������һģʽ    
  end
  er=sum(E);%����ȫ�����
  mse(cnt)=er;
  cnt=cnt+1;%����ѧϰ����
 end
 t=toc;
