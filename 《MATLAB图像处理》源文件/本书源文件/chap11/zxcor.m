function [epsilon,eta,C]=zxcor(f,D,m,n)  
%����غ���zxcor(),fΪ�����ͼ�����ݣ�DΪƫ�ƾ��룬��m��n����ͼ��ĳߴ�����
for epsilon=1:D									%ѭ�����ͼ��f(i,j)��ƫ��ֵΪD������֮������ֵ
  for eta=1:D                
     temp=0;
     fp=0;
     for x=1:m
        for y=1:n
           if(x+ epsilon -1)>m|(y+ eta -1)>n
             f1=0;
           else   
            f1=f(x,y)*f(x+ epsilon -1,y+ eta -1);     
           end
           temp=f1+temp;
           fp=f(x,y)*f(x,y)+fp;
        end      
     end 
        f2(epsilon, eta)=temp;
        f3(epsilon, eta)=fp;
        C(epsilon, eta)= f2(epsilon, eta)/ f3(epsilon, eta);						%���ֵC
   end
end
epsilon =0:(D-1);										%x�����ȡֵ��Χ
eta =0:(D-1);										%y�����ȡֵ��Χ 
