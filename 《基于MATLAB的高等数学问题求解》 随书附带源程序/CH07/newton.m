function [x,fx,iter,X]=newton(fun,x0,eps,maxiter)
%NEWTON   ţ�ٷ��󷽳̵ĸ�
% X=NEWTON(FUN,X0)  ţ�ٷ��󷽳��ڳ�ʼ��X0���ĸ�
% X=NEWTON(FUN,X0,EPS)  ţ�ٷ��󷽳��ڳ�ʼ��X0���ľ���ΪEPS�ĸ�
% X=NEWTON(FUN,X0,EPS,MAXITER)  ţ�ٷ��󷽳̵ĸ����趨����������
% [X,FX]=NEWTON(...)  ţ�ٷ�����������ظ����ĺ���ֵ
% [X,FX,ITER]=NEWTON(...)  ţ�ٷ���������ظ����ĺ���ֵ�Լ���������
% [X,FX,ITER,XS]=NEWTON(...)  ţ�ٷ���������ظ����ĺ���ֵ�����������Լ�����������
%
% ���������
%     ---FUN�����̵ĺ�������������Ϊ��������������������M�ļ���ʽ
%     ---X0����ʼ������
%     ---EPS�������趨
%     ---MAXITER������������
% ���������
%     ---X�����صķ��̵ĸ�
%     ---FX�����̸���Ӧ�ĺ���ֵ
%     ---ITER����������
%     ---XS������������
%
% See also fzero, RootInterval, bisect

if nargin<2
    error('�������������Ҫ2����')
end
if nargin<3 || isempty(eps)
    eps=1e-6;
end
if nargin<4 || isempty(maxiter)
    maxiter=1e4;
end
s=symvar(fun);
if length(s)>1
    error('����fun����ֻ����һ�����ű���.')
end
df=diff(fun,s);
k=0;err=1;
while abs(err)>eps
    k=k+1;
    fx0=subs(fun,s,x0);
    dfx0=subs(df,s,x0);
    if dfx0==0
        error('f(x)��x0���ĵ���Ϊ0��ֹͣ����')
    end
    x1=x0-fx0/dfx0;
    err=x1-x0;
    x0=x1;
    X(k)=x1;
end
if k>=maxiter
    error('�����������ޣ�����ʧ�ܣ�')
end
x=x1;fx=subs(fun,x);iter=k;X=X';
web -broswer http://www.ilovematlab.cn/forum-221-1.html