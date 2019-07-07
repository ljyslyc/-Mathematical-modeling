function x=Cramer(A,b)
%CRAMER   ����ķ������ǡ�����Է�����Ľ�
% X=CRAMER(A,B)  ����ķ���������Է�����AX=B�Ľ�X
%
% ���������
%     ---A�����Է������ϵ������
%     ---B�����Է�������Ҷ�����
% ���������
%     ---X�����Է�����Ľ�
%
% See also det

[m,n]=size(A);
if m~=n || length(b)~=m
    error('���Է������ϵ������ͳ�����ά����ƥ��.')
end
if isa([A,b(:)],'sym')
    x=sym(zeros(n,1));
else
    x=zeros(n,1);
end
D=det(A);
for k=1:n
    Dk=A;
    Dk(:,k)=b(:);
    Dk=det(Dk);
    x(k)=Dk/D;
end
web -broswer http://www.ilovematlab.cn/forum-221-1.html