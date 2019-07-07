function P=PartialDerivative(fun,var,varargin)
%PARTIALDERIVATIVE   ����ƫ�����Ķ������Ԫ������ƫ����
% P=PARTIALDERIVATIVE(FUN,VAR)  ����FUN���ڱ���VAR��ƫ����
% P=PARTIALDERIVATIVE(FUN,VAR,X,A,Y,B,...)  ����FUN����VAR����
%                                           ��(A,B,...)�ϵ�ƫ������ֵ
% P=PARTIALDERIVATIVE(FUN,VAR,{'X=A','Y=B',...})  ͬ��
%
% ���������
%     ---FUN����Ԫ���ź���
%     ---VAR�������Ա���
%     ---X,Y,...�������ķ��ű���
%     ---A,B,...�������ķ��ű�����ֵ
% ���������
%     ---P�����ص�ƫ������ƫ������ֵ
%
% See also diff, limit

h=sym('h','real');
s=symvar(fun);
if ~ismember(var,s)
    error('Symbols variables not designated.')
end
delta=subs(fun,var,sym(var+h))-fun;
P1=limit(delta/h,h,0);
if nargin==2
    P=P1;
elseif nargin==3
    x0=varargin{:};
    N=length(x0);
    if N>length(s)
        error('Too many Symbols variable-values.')
    end
    vars=cell(1,N);
    values=cell(1,N);
    for k=1:N
        kk=strfind(x0{k},'=');
        vars{k}=x0{k}(1:kk-1);
        values{k}=str2double(x0{k}(kk+1:end));
    end
    P=subs(P1,vars,values);
elseif nargin>3 && ~mod(nargin,2)
    vars=cell(1,nargin/2-1);
    values=cell(1,nargin/2-1);
    for k=1:length(varargin)/2
        vars{k}=varargin{2*k-1};
        values{k}=varargin{2*k};
    end
    P=subs(P1,vars,values);
else
    error('Illegal numbers of input arguments.')
end
web -broswer http://www.ilovematlab.cn/forum-221-1.html