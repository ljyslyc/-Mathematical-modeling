function [x,fx,iter,X]=bisect(fun,a,b,eps,varargin)
%BISECT   ���ַ��󷽳̵ĸ�
% X=BISECT(FUN,A,B)
% X=BISECT(FUN,A,B,EPS)
% X=BISECT(FUN,A,B,EPS,P1,P2,...)
% [X,FX]=BISECT(...)
% [X,FX,ITER]=BISECT(...)
% [X,FX,ITER,XS]=BISECT(...)
%
% ���������
%     ---FUN�����̵ĺ�������������Ϊ��������������������M�ļ���ʽ
%     ---A,B������˵�
%     ---EPS�������趨
%     ---P1,P2,...�����̵ĸ��Ӳ���
% ���������
%     ---X�����صķ��̵ĸ�
%     ---FX�����̸���Ӧ�ĺ���ֵ
%     ---ITER����������
%     ---XS������������
%
% See also fzero, RootInterval

if nargin<3
    error('�������������Ҫ3����')
end
if nargin<4 || isempty(eps)
    eps=1e-6;
end
fa=feval(fun,a,varargin{:});
fb=feval(fun,b,varargin{:});
% fa=fun(a,varargin{:});fb=fun(b,varargin{:});
k=1;
if fa*fb>0  % ��������ַ�ʹ������
    warning(['����[',num2str(a),',',num2str(b),']�ڿ���û�и�']);
elseif fa==0  % ������˵�Ϊ��
    x=a; fx=fa;
elseif fb==0  % �����Ҷ˵�Ϊ��
    x=b; fx=fb;
else
    while abs(b-a)>eps;  % ���ƶ��ַ���������
        x=(a+b)/2;  % ��������˵�
        fx=feval(fun,x,varargin{:}); % �����е�ĺ���ֵ
        if fa*fx>0;  % ����
            a=x;   % �˵����
            fa=fx;  % �˵㺯��ֵ����
        elseif fb*fx>0;  % ����
            b=x;  % �˵����
            fb=fx;  % �˵㺯��ֵ����
        else
            break
        end
        X(k)=x;
        k=k+1;
    end
end
iter=k;
web -broswer http://www.ilovematlab.cn/forum-221-1.html