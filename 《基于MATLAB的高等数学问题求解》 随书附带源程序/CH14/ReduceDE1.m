% function y=ReduceDE1(f,n,type)
% %REDUCEDE1   ����y^(n)=f(x)�͵�΢�ַ��̵����
% % Y=REDUCEDE1(F,N)  ���÷ǵݹ��㷨�󷽳�y^(n)=f(x)��ͨ��
% % Y=REDUCEDE1(F,N,TYPE)  ����ָ�����㷨�󷽳�y^(n)=f(x)��ͨ��
% %
% % ���������
% %     ---F��΢�ַ��̵��Ҷ˺���
% %     ---N��΢�ַ��̵Ľ�
% %     ---TYPE��ָ�����㷨���ͣ���'non-recursive'��'recursive'����ȡֵ
% % ���������
% %     ---Y��΢�ַ��̵Ľ�
% %
% % See also dsolve
% 
% if nargin==2
%     type='non-recursive';
% end
% switch lower(type)
%     case {1,'non-recursive'}
%         C=sym('C%d',[1,n]);
%         for k=1:n
%             f=int(f,'x')+C(k);
%         end
%     case {2,'recursive'}
%         syms C1
%         if n==1
%             f=int(f,'x')+C1;
%         else
%             f=ReduceDE1(sym([char(int(f,'x')),'+C',int2str(n)]),n-1,2);
%         end
% end
% y=f;
dydx=@(lambda,q)@(x,y)[y(2);
    -(lambda-2*q*cos(2*x))*y(1)];
q=5; lambda=15;
[x,y]=ode45(dydx(lambda,q),[0,pi],[1,0]);
dy=y(end,2);
dy0=0;
while abs(dy-dy0)>=1e-10
    lambda=lambda+dy-dy0;
    [x,y]=ode45(dydx(lambda,q),[0,pi],[1,0]);
    dy=y(end,2);
end
plot(x,y(:,1))
title(['��ǰ��\lambda=',num2str(lambda)])
web -broswer http://www.ilovematlab.cn/forum-221-1.html