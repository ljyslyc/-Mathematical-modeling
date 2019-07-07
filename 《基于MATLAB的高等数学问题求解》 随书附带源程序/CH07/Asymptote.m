function equation=Asymptote(fun,varargin)
%ASYMPTOTE   �����ߵĽ�����
% EQUATION=ASYMPTOTE(FUN,H)  �����ߵ�ˮƽ�����ߣ�H����Ϊ1,'h','hor'��'horizontal'
% EQUATION=ASYMPTOTE(FUN,V,X0)  �����ߵĴ�ֱ�����ߣ�V����Ϊ2,'v','ver'��'vertical'
% EQUATION=ASYMPTOTE(FUN,L)  �����ߵ�б�����ߣ�L����Ϊ3,'l'��'lean'
%
% ���������
%     ---FUN�����߷��̵ĺ�����ʽ
%     ---X0����ֱ�����߷���X=X0
%     ---H,V,L��ָ�������ߵ����ͣ�H��ʾˮƽ�����ߣ�V��ʾ��ֱ�����ߣ�L��ʾб������
% ���������
%     ---EQUATION�������ߵķ��̣��������ڣ��򷵻��ַ���'������'
%
% See also limit

type=varargin{1};
x=sym('x','real');
s=symvar(fun);
if length(s)>1
    error('����fun����ֻ����һ�����ű���.')
end
if ~isequal(x,s)
    fun=subs(fun,s,x);
end
switch lower(type)
    case {1,'h','hor','horizontal'}  % ˮƽ������
        k=limit(fun,x,inf);
        if isinf(double(k))
            equation='������';
        else
            equation=char(['y=',char(k)]);
        end
    case {2,'v','ver','vertical'}  % ��ֱ������
        x0=varargin{2};
        if isempty(x0) || nargin==2
            equation='������';
        else
            N=length(x0);
            equation=cell(1,N);
            for k=1:N
                if ~isinf(double(limit(fun,x,x0(k),'right'))) &&...
                        ~isinf(double(limit(fun,x,x0(k),'left')))
                    equation{k}='������';
                else
                    equation{k}=char(['x=',char(sym(x0(k)))]);
                end
            end
        end
    case {3,'l','lean'}  % б������
        K=limit(fun/x,x,inf);
        b=limit(fun-K*x,inf);
        if isinf(double(K)) || isequal(K,0)
            equation='������';
        else
            equation=char(['y=',char(K),'*x+',char(b)]);
        end
    otherwise
        error('Illegal options.')
end
web -broswer http://www.ilovematlab.cn/forum-221-1.html