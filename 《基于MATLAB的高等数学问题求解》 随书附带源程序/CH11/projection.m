function h=projection(F,G,limit,type)
%PROJECTION   ���ƿռ��������������ϵ�ͶӰ
% PROJECTION(F,G)  ��������F��G�Ľ�����xOy����x��[-2*pi,2*pi]��ͶӰ
% PROJECTION(F,G,LIMIT)  ��������F��G�Ľ�����xOy����x��LIMIT��ͶӰ
% PROJECTION(F,G,LIMIT,TYPE)  ��������F��G�Ľ�����ָ���������ϵ�ͶӰ
% H=PROJECTION(...)  ���ƿռ��������������ϵ�ͶӰ����������
%
% ���������
%     ---F,G�������ཻ�����淽��
%     ---LIMIT���Ա�����Χ
%     ---TYPE��ָ�������������
% ���������
%     ---H��ͶӰͼ�εľ��
%
% See also ezplot

if nargin<4
    type='z';
end
if nargin<3
    limit=[-2*pi,2*pi];
end
s=unique([symvar(F),symvar(G)]);
if ~ismember(type,s)
    error('Illegal options.')
end
x=solve(F,type);
G=subs(G,type,x(1));
hp=ezplot(G,limit);
if nargout>0
    h=hp;
end
web -broswer http://www.ilovematlab.cn/forum-221-1.html