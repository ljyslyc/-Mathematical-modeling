function r=RootInterval(fun,a,b,h)
%ROOTINTERVAL   ��ɨ�跨�󷽳̵ĸ�������
% R=ROOTINTERVAL(FUN,A,B)
% R=ROOTINTERVAL(FUN,A,B,H)
%
% ���������
%     ---FUN�����̵�MATLAB����������Ϊ������������������
%     ---A,B������˵�
%     ---H������
% ���������
%     ---R�����صĸ������䣬��һ������Ϊ2�ľ���

if nargin==3
    h=(b-a)/100;
end
a1=a;b1=a1+h;
r=[];
while b1<b
    if fun(a1)*fun(b1)<0
        r=[r;[a1,b1]];
        a1=b1;b1=a1+h;
    else
        a1=b1;b1=a1+h;
        continue
    end
end
web -broswer http://www.ilovematlab.cn/forum-221-1.html