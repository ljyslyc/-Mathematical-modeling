function [x,L,U]=LU_Equ(A,b)
%LU_EQU   LU�ֽⷨ�����Է�����Ľ�
% X=LU_EQU(A,B)  LU�ֽⷨ�����Է�����AX=B�Ľ�X
% [X,L,U]=LU_EQU(A,B)  LU�ֽⷨ�����Է�����AX=B�Ľ�X�������طֽ�����(��)���Ǿ���
%
% ���������
%     ---A�����Է������ϵ������
%     ---B�����Է�������Ҷ���
% ���������
%     ---X�����Է�����Ľ�
%     ---L���ֽ��������Ǿ���
%     ---U���ֽ��������Ǿ���
%
% See also TriuEqu

[m,n]=size(A);
if m~=n || length(b)~=m
    error('���Է������ϵ������ͳ�����ά����ƥ��.')
end
if isa([A,b(:)],'sym')
    U=sym(zeros(n));
    L=sym(eye(n));
else
    U=zeros(n);
    L=eye(n);
end
U(1,:)=A(1,:);
L(2:n,1)=A(2:n,1)/U(1,1);
for k=2:n
    U(k,k:n)=A(k,k:n)-L(k,1:k-1)*U(1:k-1,k:n);
    L(k+1:n,k)=A(k+1:n,k)-L(k+1:n,1:k-1)*U(1:k-1,k);
    L(k+1:n,k)=L(k+1:n,k)/U(k,k);
end
y=flipud(TriuEqu(rot90(L,2),flipud(b(:))));
x=TriuEqu(U,y);
web -broswer http://www.ilovematlab.cn/forum-221-1.html