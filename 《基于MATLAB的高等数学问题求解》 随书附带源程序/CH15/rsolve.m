function x=rsolve(F,G,u,x0)
%RSOLVE   z�任�����ɢ���Զ���ϵͳ
% X=RSOLVE(F,G,U,X0)  �����Զ���ϵͳX(k+1)=F*X(k)+G*U(k)�Ľ�
%
% ���������
%     ---F,G��ϵͳ��ϵ������
%     ---U��ϵͳ����
%     ---X0��ϵͳ�ĳ�ʼֵ
% ���������
%     ---X��ϵͳ�Ľ�
%
% See also ztrans, iztrans

[m,n]=size(F);
[q,p]=size(G);
r=length(u);
if m~=n || n~=q
    error('ϵ������ά����ƥ��.')
end
if isvector(u)
    if r~=p
        error('������������ƾ���ά����ƥ��.')
    end
end
I=sym(eye(size(F)));
syms z k
U=ztrans(sym(u));
x=simple(iztrans((z*I-sym(F))\(z*sym(x0)+sym(G)*U),z,k));
web -broswer http://www.ilovematlab.cn/forum-221-1.html