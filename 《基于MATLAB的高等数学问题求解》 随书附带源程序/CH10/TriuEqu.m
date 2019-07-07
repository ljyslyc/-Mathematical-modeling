function x=TriuEqu(U,b)
%TRIUEQU   ��ȥ���������Ƿ�����Ľ�
% X=TRIUEQU(U,B)  ��ȥ���󷽳���UX=B�Ľ⣬����U����������
%
% ���������
%     ---U�����Է������ϵ��������һ�������Ǿ���
%     ---B�����Է�������Ҷ�����
% ���������
%     ---X�����Է�����Ľ�
%
% See also Cramer

[m,n]=size(U);
if m~=n || length(b)~=m
    error('���Է������ϵ������ͳ�����ά����ƥ��.')
end
if isa([U,b(:)],'sym')
    x=sym(zeros(n,1));
else
    x=zeros(n,1);
end
x(n)=b(n)/U(n,n);  % ��x_n
for k=n-1:-1:1
    x(k)=(b(k)-U(k,k+1:n)*x(k+1:n))/U(k,k);  % ��x_k��k=n-1,n-2,��,1
end
web -broswer http://www.ilovematlab.cn/forum-221-1.html