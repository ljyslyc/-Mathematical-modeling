function y=realfunvalue(fun,x)
%REALFUNVALUE   �ڸ�����Χ��������ĳ�㴦��ʵ����ֵ
% Y=REALFUNVALUE(FUN,X)  �ڸ�����Χ������FUN��X����ʵ����ֵ
%
% ���������
%     ---FUN�������ķ��ű��ʽ
%     ---X��ָ�����Ա���ֵ
% ���������
%     ---Y�����ص�ʵ����ֵ
%
% See also finverse, solve

warning off all
F=subs(fun,x);
if ~isreal(F)
    t=symvar(fun);
    t=sym(t,'real');
    f=finverse(fun);
    y=solve(f-x,t);
else
    y=F;
end
y=double(y);
web -broswer http://www.ilovematlab.cn/forum-221-1.html