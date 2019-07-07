function [X,FVAL,TYPE]=Extremum(fun,range)
%EXTREMUM   ������ָ�������ϵļ�ֵ�㼰��ֵ
% X=EXTREMUM(FUN,RANGE)
% [X,FVAL]=EXTREMUM(...)
% [X,FVAL,TYPE]=EXTREMUM(...)
%
% ���������
%     ---FUN�������ķ��ű��ʽ
%     ---RANGE����ֵ����
% ���������
%     ---X����ֵ��
%     ---FVAL����ֵ
%     ---TYPE����ֵ��������
%
% See also diff, solve

x=sym('x','real');
s=symvar(fun);
if length(s)>1
    error('����fun����ֻ����һ�����ű���.')
end
if ~isequal(x,s)
    fun=subs(fun,s,x);
end
df=diff(fun);
x0=unique(double(solve(df)));
d2f=diff(df);
N=length(x0);
X=[];
for kk=1:N
    if prod(x0(kk)-range)<=0
        X=[X,x0(kk)];
    end
end
FVAL=subs(fun,X);
D=subs(d2f,X);
TYPE=cell(1,N);
for k=1:N
    if D(k)==0
        TYPE{k}='��ȷ��';
    elseif D(k)>0
        TYPE{k}='��Сֵ';
    else
        TYPE{k}='����ֵ';
    end
end
web -broswer http://www.ilovematlab.cn/forum-221-1.html