function S=LineConvert(PI1,PI2)
%LINECONVERT   ��ֱ�ߵ�һ�㷽��ת��Ϊ��������
% S=LINECONVERT(PI1,PI2)  ��ƽ��PI1��PI2�Ľ��ߵĲ�������
%
% ���������
%     ---PI1,PI2��ƽ���ϵ������
% ���������
%     ---S���������̱��ʽ
%
% See also \, cross

if ~isvector(PI1) && ~isvector(PI2)
    error('PI1 and PI2 must be vectors.')
end
if length(PI1)==4 && length(PI2)==4
    A=[PI1(1:3);PI2(1:3)];
    b=-[PI1(4);PI2(4)];
    x0=A\b;
    s=cross(A(1,:),A(2,:));
    syms t
    S=x0(:)+s(:)*t;
else
    error('������������Ϊ4ά����.')
end
web -broswer http://www.ilovematlab.cn/forum-221-1.html