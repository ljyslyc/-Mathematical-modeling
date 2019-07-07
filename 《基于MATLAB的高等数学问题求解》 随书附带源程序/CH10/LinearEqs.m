function [x,e]=LinearEqs(A,b)
%LINEAREQS   ���Է���������
% X=LINEAREQS(A,B)  ����MATLAB�Դ����������Է�����AX=B�Ľ�X
% [X,E]=LINEAREQS(A,B)  �����Է�����Ľ�X�����������
%
% ���������
%     ---A���������ϵ������
%     ---B����������Ҷ�����
% ���������
%     ---X��������Ľ�
%     ---E��������
%
% See also null, inv(\), pinv

[m,n]=size(A);b=b(:);
if m~=length(b);
    error('ϵ������A���Ҷ�����bά����ƥ��.')
end
r1=rank(A);r2=rank([A b]);
if ~all(b)  % ������Է�����
    if r1==n
        x=zeros(size(b));
    else
        z=null(sym(A));   %����淶���Ļ���ռ�
        k=sym('k%d',[n-r1,1]);   %�����������ϵ��Ӧ��ϵ��
        x=z*k;   %ԭ���̵�ͨ��
    end
else  % ��������Է�����
    if r1==r2&&r1==n
        disp('��������ǡ���ģ���Ψһ�⣡')
        x=A\b;
    elseif r1==r2&&r1~=n
        disp('��������Ƿ���ģ�������⣡')
        warning off all
        z=null(sym(A));   %����淶���Ļ���ռ�
        x0=sym(A)\b;  %���һ���ؽ�
        k=sym('k%d',[n-r1,1]);   %�����������ϵ��Ӧ��ϵ��
        x=x0+z*k;   %ԭ���̵�ͨ��
    else
        disp('�������ǳ����ģ�ֻ����С���������µĽ⣡')
        x=pinv(A)*b;
    end
end
e=norm(double(A*x-b));
web -broswer http://www.ilovematlab.cn/forum-221-1.html