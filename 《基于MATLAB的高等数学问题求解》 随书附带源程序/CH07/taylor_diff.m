function T=taylor_diff(fx,n,x,x0)
%TAYLOR_DIFF   ����̩�չ�ʽ�Ķ���ʽ������̩��չ��ʽ
% T=TAYLOR_DIFF(F)
% T=TAYLOR_DIFF(F,N)
% T=TAYLOR_DIFF(F,N,X)
% T=TAYLOR_DIFF(F,N,X,X0)
%
% ���������
%     ---F�������ķ��ű��ʽ
%     ---N��̩��չ��ʽ�Ľ״�
%     ---X�������Ա���
%     ---X0��̩��չ����
% ���������
%     ---T�����ص�̩��չ��ʽ
%
% See also diff, limit

if nargin<4
    x0=0;
end
if nargin<3
    x=symvar(fx);
    if length(x)>1
        error('The Symbolic variable not point out.')
    end
end
if nargin<2
    n=6;
end
a=cell(1,n);
T=limit(fx,x,x0);
for k=2:n
    a{k}=1/sym(factorial(k-1))*limit(diff(fx,x,k-1),x,x0);
    T=T+a{k}*(x-x0)^(k-1);
end
web -broswer http://www.ilovematlab.cn/forum-221-1.html