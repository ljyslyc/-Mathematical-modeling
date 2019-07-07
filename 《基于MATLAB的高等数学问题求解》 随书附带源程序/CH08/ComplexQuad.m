function [I,str]=ComplexQuad(varargin)
%COMPLEXQUAD   �������������ⶨ����
% I=COMPLEXQUAD(X,Y,TYPE)  ʹ��ָ���ĸ��������������ɢ���ݵ���ֵ����
% I=COMPLEXQUAD(FUN,A,B,N,TYPE)  ʹ��ָ���ĸ��������������FUN����ֵ����
% [I,STR]=COMPLEXQUAD(...)  ʹ�ø��������������ֵ���ֲ����������õĸ�������
%
% ���������
%     ---X,Y���۲����ݣ��ȳ�����
%     ---FUN����������
%     ---A,B���������޺�����
%     ---N����������ȷ���
%     ---TYPE��ָ���ĸ����������ͣ�������ȡֵ��
%              1.'trape'��1�������������
%              2.'simpson'��2����������ɭ���
%              3.'cotes'��4������Cotes���
% ���������
%     ---I�����ص���ֵ����ֵ
%     ---STR�����صĸ�������
%
% See also InterpolatoryQuad

args=varargin;
type=args{end};
num=[1,2,4];
S={'trape','simpson','cotes'};
if ~isnumeric(type)
    I=ismember(S,type);
    n=num(I==1);
else
    n=type;
end
if isnumeric(args{1})
    x=args{1};
    y=args{2};
    N=length(x);
    if rem(N-1,n)~=0
        error('���ݵĳ�������ѡ�����������ƥ��.')
    end
    Nn=(N-1)/n;
    h=(x(N)-x(1))/Nn;
else
    [fun,a,b,Nn]=deal(args{1:end-1});
    h=(b-a)/Nn;
    x=a+h/n*(0:n*Nn);
    N=length(x);
    y=feval(fun,x);
end
switch lower(type)
    case {1,'trape'}
        str='�����������';
        I=h*[1,2*ones(1,Nn-1),1]*y'/2;
    case {2,'simpson'}
        str='��������ɭ���';
        a=[1,reshape([4*ones(1,Nn-1);2*ones(1,Nn-1)],1,[]),4,1];
        I=h/6*a*y';
    case {4,'cotes'}
        str='����Cotes���';
        a=[7,reshape([32*ones(1,Nn-1);12*ones(1,Nn-1);...
            32*ones(1,Nn-1);14*ones(1,Nn-1)],1,N-5),32,12,32,7];
        I=h/90*a*y';
    otherwise
        error('Illegal options.')
end
web -broswer http://www.ilovematlab.cn/forum-221-1.html