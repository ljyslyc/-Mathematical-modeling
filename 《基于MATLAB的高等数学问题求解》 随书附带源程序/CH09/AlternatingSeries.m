function type=AlternatingSeries(un)
%ALTERNATINGSERIES   ������������
% TYPE=ALTERNATINGSERIES(UN)  ����������Ķ����жϽ�����(-1)^(N-1)*UN����ɢ��
%
% ���������
%     ---UN�������
% ���������
%     ---TYPE������������ɢ�Ե��ַ���
%
% See also limit

n=sym('n','positive');
s=symvar(un);
if ~ismember(n,s)
    error('�����һ����ķ��ű�������Ϊn.')
end
uN=subs(un,n,n+1);
x=subs(un-uN,n,1:1e6);
L=limit(un,n,inf);
if L==0 && all(x>=0)
    type='����';
elseif L~=0
    type='��ɢ';
else
    type='��ȷ��';
end
web -broswer http://www.ilovematlab.cn/forum-221-1.html