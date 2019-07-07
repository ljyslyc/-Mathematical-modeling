function result=diff_para(y,x,t,n)
%DIFF_PARA   ����������
% R=DIFF_PARA(Y,X)��R=DIFF_PARA(Y,X,[])  �����ű��ʽXֻ����һ�����ű���ʱ��
%                                                ����X��Y�����Ĳ������̵ĵ���dY/dX
% R=DIFF_PARA(Y,X,T)  ����X��Y�����Ĳ������̹����Ա���T�ĵ���dY/dX
% R=DIFF_PARA(Y,X,T,N)  ����X��Y�����Ĳ������̹����Ա���T��N�׵���dNY/dXN
%
% ���������
%     ---Y,X���������̵ķ��ű��ʽ
%     ---T���������̵ķ����Ա���
%     ---N���󵼽״�
% ���������
%     ---R�����������󵼽��
%
% See also diff

if nargin<4
    n=1;
end
if nargin==2 || isempty(t)
    t=symvar(x);
    if length(t)>1
        error('The Symbolic variable not point out.')
    end
end
if n==1
    result=diff(y,t)/diff(x,t);
else
    result=diff(diff_para(y,x,t,n-1),t)/diff(x,t);
end
web -broswer http://www.ilovematlab.cn/forum-221-1.html