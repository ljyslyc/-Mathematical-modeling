function [Interval,type]=Monotonicity(varargin)
%MONOTONICITY   ������ָ�������ϵĵ������估������
% [INTERVAL,TYPE]=MONOTONICITY(FUN,DOMAIN)  ����FUN������DOMAIN�ϵĵ�������
% [INTERVAL,TYPE]=MONOTONICITY(FUN,DOMAIN,X0)  ����FUN������DOMAIN�ϵĵ���
%                      ���䣬���к���FUNһ�׵�����û�ж���ĵ㣨������⣩��֪
%
% ���������
%     ---FUN���������ʽ
%     ---DOMAIN��ָ��������
%     ---X0��һ�׵�����û�ж���ĵ㣨������⣩
% ���������
%     ---INTERVAL�����صĵ�������
%     ---TYPE�������������Ϻ����ĵ�����
%
% See also solve, diff

warning off all
[fun,domain]=deal(varargin{1:2});
x=sym('x','real');
s=symvar(fun);
if length(s)>1
    error('����fun����ֻ����һ�����ű���.')
end
if ~isequal(x,s)
    fun=subs(fun,s,x);
end
df=diff(fun);
[num,den]=numden(df);
xd=solve(den);
xd=double(xd);
x=solve(num);
x=double(x);
x=unique([xd(:);x(:)]);
if nargin==3
    x0=varargin{3};
    x=unique([x;x0(:)]);
end
N=length(x);
Interval=cell(1,N+1);
type=cell(1,N+1);
if ~isequal(domain(1),x(1))
    Interval{1}=[domain(1),x(1)];
    if isinf(domain(1))
        f1=realfunvalue(df,x(1)-0.1);
    else
        f1=realfunvalue(df,(domain(1)+x(1))/2);
    end
    type{1}=Judgment(f1,{'��������','��������'});
else
    Interval{1}=[];
    type{1}=[];
end
for k=2:N
    Interval{k}=[x(k-1),x(k)];
    f=realfunvalue(df,sum(x([k-1,k]))/2);
    type{k}=Judgment(f,{'��������','��������'});
end
if ~isequal(x(end),domain(2))
    Interval{end}=[x(end),domain(2)];
    if isinf(domain(2))
        f2=realfunvalue(df,x(N)+0.1);
    else
        f2=realfunvalue(df,(x(N)+domain(2))/2);
    end
    type{N+1}=Judgment(f2,{'��������','��������'});
else
    Interval{N+1}=[];
    type{N+1}=[];
end
web -broswer http://www.ilovematlab.cn/forum-221-1.html