function [x,fval,iter,exitflag]=Newtons(fun,x0,eps,maxiter)
%NEWTONS   ţ�ٷ�������Է�����ĸ�
% X=NEWTONS(FUN,X0)  ţ�ٷ�������Է�����Ľ⣬��ʼ������ΪX0
% X=NEWTONS(FUN,X0,EPS)  ţ�ٷ�������Է�����Ľ⣬����Ҫ��ΪEPS
% X=NEWTONS(FUN,X0,EPS,MAXITER)  ţ�ٷ�������Է�����Ľ⣬����������ΪMAXITER
% [X,FVAL]=NEWTONS(...)  ţ�ٷ�������Է�����ĽⲢ���ؽ⴦�ĺ���ֵ
% [X,FVAL,ITER]=NEWTONS(...)  ţ�ٷ�������Է�����ĽⲢ���ص�������
% [X,FVAL,ITER,EXITFLAG]=NEWTONS(...)  ţ�ٷ�������Է�����ĽⲢ���ص����ɹ���־
%
% ���������
%     ---FUN�������Է�����ķ��ű��ʽ
%     ---X0����ʼ����������
%     ---EPS������Ҫ��Ĭ��ֵΪ1e-6
%     ---MAXITER��������������Ĭ��ֵΪ1e4
% ���������
%     ---X�������Է��̵Ľ��ƽ�����
%     ---FVAL���⴦�ĺ���ֵ
%     ---ITER����������
%     ---EXITFLAG�������ɹ���־��1��ʾ�ɹ���0��ʾʧ��
%
% See also newton

if nargin<2
    error('�������������Ҫ2��.')
end
if nargin<3
    eps=1e-6;
end
if nargin<4
    maxiter=1e4;
end
if isa(fun,'inline')
    fun=char(fun);
    k=strfind(fun,'.');
    fun(k)=[];
    fun=sym(fun);
elseif ~isa(fun,'sym')
    error('�������ͱ�����������������ź���.')
end
s=symvar(fun);
if length(s)>length(x0)
    error('���������ɱ�������.')
end
x0=x0(:);
J=jacobian(fun,s);
k=0;err=1;
exitflag=1;
while err>eps
    k=k+1;
    fx0=subs(fun,num2cell(s),num2cell(x0));
    J0=subs(J,num2cell(s),num2cell(x0));
    x1=x0-J0\fx0;
    err=norm(x1-x0);
    x0=x1;
    if k>=maxiter
        exitflag=0;
        break
    end
end
x=x1;
fval=fx0;
iter=k;
web -broswer http://www.ilovematlab.cn/forum-221-1.html