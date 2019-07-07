function [A,B,F] = fseriesquadl(fun,x,n,a,b)
%FSERIESQUADL   ����Ҷ��������ֵ���
% [A,B,F]=FSERIESQUADL(FUN,X,N)  ����FUN������[-pi,pi]�ϵ�N����ֵ����Ҷչʽ
% [A,B,F]=FSERIESQUADL(FUN,X,N,ALPHA,BETA)  ����FUN��ָ�������ϵ���ֵ����Ҷչʽ
%
% ���������
%     ---FUN�������Ĵ�չ������
%     ---X���Ա�������
%     ---N��չ������
%     ---ALPHA,BETA������չ�����䣬Ĭ��ֵΪ[-pi,pi]
% ���������
%     ---A,B������Ҷϵ������
%     ---F�������ĸ���Ҷչ��ʽ��X�ϵ�ֵ
%
% See also quadl, fseriessym

if nargin==3
    a=-pi;b=pi; 
end
L=(b-a)/2;
f=inline(fun);
var=char(argnames(f));
A=zeros(1,n+1);B=zeros(1,n);
A(1) = quadl(f,-L,L)/L; % ����A_0
F=A(1)/2;
for k=1:n;
    fcos=inline(['(',fun,')','.*cos(',num2str(k*pi/L),'*',var,')']); 
    fsin=inline(['(',fun,')','.*sin(',num2str(k*pi/L),'*',var,')']); 
    A(k+1) =quadl(fcos,-L,L)/L;  % ����ϵ��A(2:n+1)
    B(k)=quadl(fsin,-L,L)/L;  % ����ϵ��B(1:n)
    F=F+A(k+1)*cos(k*pi*x/L)+B(k)*sin(k*pi*x/L);
end
web -broswer http://www.ilovematlab.cn/forum-221-1.html