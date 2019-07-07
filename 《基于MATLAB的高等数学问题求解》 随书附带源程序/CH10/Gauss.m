function [x,U]=Gauss(A,b)
%GAUSS   ��˹��ȥ�������Է�����Ľ�
% X=GAUSS(A,B)  ��˹��ȥ�������Է�����AX=B�Ľ�X
% [X,U]=GAUSS(A,B)  ��˹��ȥ�������Է�����AX=B�Ľ�X��������Ԫ��������Ƿ�������������
%
% ���������
%     ---A�����Է������ϵ������
%     ---B�����Է�������Ҷ���
% ���������
%     ---X�����Է�����Ľ�
%     ---U����Ԫ��������Ƿ�������������
%
% See also TriuEqu

[m,n]=size(A);
if m~=n || length(b)~=m
    error('���Է������ϵ������ͳ�����ά����ƥ��.')
end
% ��Ԫ����
for k=1:n-1
    m=A(k+1:n,k)/A(k,k);
    A(k+1:n,k+1:n)=A(k+1:n,k+1:n)-m*A(k,k+1:n);
    b(k+1:n)=b(k+1:n)-m*b(k);
    if isa([A,b(:)],'sym')
        A(k+1:n,k)=sym(zeros(n-k,1));
    else
        A(k+1:n,k)=zeros(n-k,1);
    end
end
U=[A,b];
x=TriuEqu(A,b);
web -broswer http://www.ilovematlab.cn/forum-221-1.html