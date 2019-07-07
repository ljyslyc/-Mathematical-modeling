function [L,type]=PositiveIermSeries(un,mode)
%POSITIVEIERMSERIES   ������ı�ֵ�������͸�ֵ������
% L=POSITIVEIERMSERIES(UN)  ��ֵ�������ж����������ɢ��
% L=POSITIVEIERMSERIES(UN,MODE)  ѡ��ָ�����������ж����������ɢ��
% [L,TYPE]=POSITIVEIERMSERIES(...)  ѡ��ָ�����������ж����������ɢ��
%                                   ��������ʹ�õ�������
%
% ���������
%     ---UN�������ͨ��
%     ---MODE��ָ������������MODE����������ȡֵ��
%              1.'d'��'��ֵ'��1����ֵ������
%              2.'k'��'��ֵ'��2����ֵ������
% ���������
%     ---L�����ص�ͨ���ĳ�����͵ļ���ֵ
%     ---TYPE����ʹ�õ�������
%
% See also limit

if nargin==1
    mode=1;
end
n=sym('n','positive');
s=symvar(un);
if ~ismember(n,s)
    error('�����һ����ķ��ű�������Ϊn.')
end
switch lower(mode)
    case {1,'d','��ֵ'}
        type='��ֵ������';
        uN=subs(un,'n',n+1);
        L=limit(simple(uN/un),'n',inf);
    case {2,'k','��ֵ'}
        type='��ֵ������';
        L=limit(simple(un^(1/n)),'n',inf);
    otherwise
        error('Illegal options.')
end
if length(s)==1
    if double(L)<1
        type=[type,'������'];
    elseif double(L)>1
        type=[type,'����ɢ'];
    else
        error('��ǰ��ѡ���������ʧЧ.')
    end
end
web -broswer http://www.ilovematlab.cn/forum-221-1.html