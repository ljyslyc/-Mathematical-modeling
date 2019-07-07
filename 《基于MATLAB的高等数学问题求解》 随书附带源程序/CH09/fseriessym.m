function [A,B,F]=fseriessym(f,x,n,a,b)
%FSERIESSYM   ����Ҷ�����ķ������
% [A,B,F]=FSERIESSYM(FUN,X,N)  ������FUN������[-pi,pi]��չ��N�׸���Ҷ����
% [A,B,F]=FSERIESSYM(FUN,X,N,ALPHA,BETA)  ������FUN��ָ��������չ��N�׸���Ҷ����
%
% ���������
%     ---FUN�������Ĵ�չ������
%     ---X���Ա���
%     ---N��չ������
%     ---ALPHA,BETA������չ�����䣬Ĭ��ֵΪ[-pi,pi]
% ���������
%     ---A,B������Ҷϵ������
%     ---F�������ĸ���Ҷչ��ʽ
%
% See also int

if nargin==3
    a=-pi;b=pi; 
end
L=(b-a)/2; 
A=int(f,x,-L,L)/L;
B=[];F=A/2;
for k=1:n
   ak=int(f*cos(k*pi*x/L),x,-L,L)/L;
   bk=int(f*sin(k*pi*x/L),x,-L,L)/L;
   A=[A,ak];
   B=[B,bk];
   F=F+ak*cos(k*pi*x/L)+bk*sin(k*pi*x/L);
end
web -broswer http://www.ilovematlab.cn/forum-221-1.html