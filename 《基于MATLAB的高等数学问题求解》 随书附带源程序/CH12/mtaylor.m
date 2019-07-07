function T=mtaylor(fun,x0,n)
%MTAYLOR   ��Ԫ������̩��չ��ʽ
% T=MTAYLOR(FUN)  ���Ԫ����FUN��ԭ�㴦��6��̩��չ��ʽ
% T=MTAYLOR(FUN,X0)  ���Ԫ����FUN�ڵ�X0����6��̩��չ��ʽ
% T=MTAYLOR(FUN,X0,N)  ���Ԫ����FUN�ڵ�X0����N��̩��չ��ʽ
%
% ���������
%     ---FUN�������Ķ�Ԫ����
%     ---X0��̩��չ���㣬��Ԫ�������������������{'x=0','y=0'}
%     ---N��̩��չ���״�
% ���������
%     ---T�����ص�̩��չ��ʽ
%
% See also taylor, diff

if nargin<3
    n=6;
end
if nargin<2 || isempty(x0)
    x0={'x=0','y=0'};
end
vars=cell(1,2); values=cell(1,2);
for k=1:2
    kk=strfind(x0{k},'=');
    vars{k}=x0{k}(1:kk-1);
    values{k}=sym(x0{k}(kk+1:end));
end
T=subs(fun,vars,values);
for m=1:n
    S=0;
    for p=0:m
        sigma=nchoosek(m,p)*(sym(vars{1})-values{1})^p*...
            (sym(vars{2})-values{2})^(m-p)*...
            subs(diff(diff(fun,vars{1},p),vars{2},m-p),vars,values);
        S=S+sigma;
    end
    T=T+S/gamma(m+1);
end
web -broswer http://www.ilovematlab.cn/forum-221-1.html