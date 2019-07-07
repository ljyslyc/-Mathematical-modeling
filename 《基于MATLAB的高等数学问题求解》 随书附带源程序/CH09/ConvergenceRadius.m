function [R,D]=ConvergenceRadius(an)
%CONVERGENCERADIUS   �ݼ����������뾶��������
% R=CONVERGENCERADIUS(AN)  ���ݼ���AN�������뾶
% [R,D]=CONVERGENCERADIUS(AN)  ���ݼ���AN�������뾶��������
%
% ���������
%     ---AN���ݼ���һ����
% ���������
%     ---R�������뾶
%     ---D��������
%
% See also limit

n=sym('n','positive');
s=symvar(an);
if ~ismember(n,s)
    error('�ݼ���ϵ���ķ��ű�������Ϊn.')
end
aN=subs(an,n,n+1);
rho=limit(simple(abs(aN/an)),n,inf);
R=1/rho;
if R==0
    D=0;
elseif isinf(double(R))
    D='(-��,+��)';
else
    D=[-R,R];
end
web -broswer http://www.ilovematlab.cn/forum-221-1.html