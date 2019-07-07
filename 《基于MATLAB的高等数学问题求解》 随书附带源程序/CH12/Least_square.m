function [p,A,b,FitFun]=Least_square(x,y,phifun,wfun)
%LEAST_SQUARE   ��С���˷�
% P=LEAST_SQUARE(X,Y,PHIFUN)  ���û�����PHIFUN���������X��Y
% P=LEAST_SQUARE(X,Y,PHIFUN,WFUN)  ���û�����PHIFUN��Ȩ����WFUN�������X��Y
% [P,A,B]=LEAST_SQUARE(...)  ��С���˷�������ݲ����ط��������ϵ��������Ҷ�����
% [P,A,B,FITFUN]=LEAST_SQUARE(...)  ��С���˷�������ݲ�������Ϻ������ʽ
%
% ���������
%     ---X,Y��ʵ������
%     ---PHIFUN����ϻ�������������������������������
%     ---WFUN��Ȩ����
% ���������
%     ---P����С�������ϵ��
%     ---A�����������ϵ������
%     ---B������������Ҷ�����
%     ---FITFUN����Ϻ������ʽ
%
% See also polyfit, lsqcurvefit

x=x(:); y=y(:);
if length(x)~=length(y)
    error('ʵ�����ݳ��Ȳ�һ��.')
end
if nargin<4
    wfun=ones(size(x));
end
func=char(phifun);
func=strrep(func,'ones(size(x))','1');
func=strrep(func,'.*','*');
func=strrep(func,'./','/');
func=strrep(func,'.^','^');
k=strfind(func,'[');
func=sym(func(k(1):end));
phifun=phifun(x);
n=size(phifun,2);
A=zeros(n);b=zeros(n,1);
for k=1:n
    for j=1:n
        A(k,j)=0;
        for i=1:length(x)
            A(k,j)=A(k,j)+wfun(i)*phifun(i,j)*phifun(i,k);
        end
    end
    for i=1:length(x)
        b(k)=b(k)+wfun(i)*y(i)*phifun(i,k);
    end
end
p=A\b;
FitFun=vpa(dot(p,func),4);
web -broswer http://www.ilovematlab.cn/forum-221-1.html