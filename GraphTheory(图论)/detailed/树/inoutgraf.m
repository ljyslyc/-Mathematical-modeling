function [ k1 k2 ] = inoutgraf( A,k)
%����ͼ������������������Ŀ���㷨
% A��ʾͼ�Ĺ�������
% k1��ʾ����������Ŀ
% k2��ʾ����������Ŀ
n = size(A,1);
c=[1:k-1,k+1:n];
C=A(c,:);
D = C;
D(D>0)=0;
B=C*D';
k1=det(B);
E=C;
E(E<0)=0;
B = C*E';
k2=det(B);
end

