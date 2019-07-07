function [L,type]=LimitSeries(un,p)
%LIMITSERIES   ������ļ���������
% L=LIMITSERIES(UN)  �����������ж������UN����ɢ�ԣ�p-������pȡ1
% L=LIMITSERIES(UN,P)  �����������ж������UN����ɢ�ԣ�P>1
% [L,TYPE]=LIMITSERIES(...)  �����������ж������UN����ɢ�ԣ������ؼ�������ɢ���ַ���
%
% ���������
%     ---UN�������
%     ---P��p�����Ľ״�
% ���������
%     ---L������ֵ
%     ---TYPE������������ɢ�Ե��ַ���
%
% See also limit

if nargin==1
    p=1;
end
if p<1
    error('�ȱȼ�������ָ��p������ڵ���1.')
end
n=sym('n','positive');
s=symvar(un);
if ~ismember(n,s)
    error('�����һ����ķ��ű�������Ϊn.')
end
L=limit(n^p*un,n,inf);
if p==1
    if length(s)==1
        if double(L)>0
            type='��ɢ';
        end
    end
else
    if double(L)>=0
        type='����';
    end
end
web -broswer http://www.ilovematlab.cn/forum-221-1.html