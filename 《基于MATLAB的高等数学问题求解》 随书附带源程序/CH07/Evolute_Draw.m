function Evolute_Draw(varargin)
%EVOLUTE_DRAW   �������ߵĽ�����
% EVOLUTE_DRAW(FUN,RANGE)
% EVOLUTE_DRAW(FUNX,FUNY,RANGE)
%
% ���������
%     ---FUN�����ߺ�����һ�㷽��
%     ---FUNX,FUNY�����ߺ����Ĳ�������
%     ---RANGE����ͼ����
%
% See also diff, diff_para

args=varargin;
range=args{end};
if nargin==2
    y=args{1};
    x=sym('x','real');
    s=symvar(y);
    if length(s)>1
        error('����fun����ֻ����һ�����ű���.')
    end
    if ~isequal(x,s)
        y=subs(fun,s,x);
    end
    df=simple(diff(y,x));
    d2f=simple(diff(df,x));
elseif nargin==3
    x=args{1}; y=args{2};
    t=sym('t','real');
    s=unique([symvar(x),symvar(y)]);
    if length(s)>1
        error('����funx��funy����ֻ����һ�����ű���.')
    end
    if ~isequal(t,s)
        x=subs(x,s,t);
        y=subs(y,s,t);
    end
    df=simple(diff_para(y,x,t,1));
    d2f=simple(diff_para(y,x,t,2));
else
    error('The number of input arguments is illegal.')
end
X=inline(x);
Y=inline(y);
alpha=inline(simple(x-df*(1+df^2)/d2f));
beta=inline(simple(y+(1+df^2)/d2f));
theta=linspace(range(1),range(2),300);
plot(X(theta),Y(theta),'k',alpha(theta),beta(theta),'r--')
xx=sort([X(range),alpha(range)]);
xlim([xx(1) xx(end)])
web -broswer http://www.ilovematlab.cn/forum-221-1.html