function varargout=max_min(fun,xrange,yrange,type)
%MAX_MIN   ��֤�н�������϶�Ԫ�����Ľ�ֵ����
% MAX_MIN(FUN,XRANGE,YRANGE)  ͼ�λ�������ʾ�н�������϶�Ԫ�����Ľ�ֵ����
% MAX_MIN(FUN,XRANGE,YRANGE,TYPE)  ͼ�λ�������ʾ�н�������϶�Ԫ�����Ľ�ֵ����
%                                  ͼ�ο�����������ʾ��ʽ��'rect'��'circ'
% [ZMAX,ZMIN]=MAX_MIN(...)  ���غ�����ָ�������ϵ����ֵ����Сֵ
%
% ���������
%     ---FUN�������Ķ�Ԫ����
%     ---XRANGE,YRANGE���Ա�����Χ
%     ---TYPE��ͼ�λ������ͣ���'rect'��'circ'����ȡֵ
% ���������
%     ---ZMAX,ZMIN�����������ֵ����Сֵ
%
% See also ezsurf, max, min

if nargin==3
    type='circ';
end
if ~any(strcmp(type,{'rect','circ'}))
    error('The Input argument type must be either ''rect'' or ''circ''.')
end
h=ezsurf(fun,[xrange yrange],type);
X=get(h,'XData');
Y=get(h,'YData');
Z=get(h,'ZData');
zmax=max(Z(:));
zmin=min(Z(:));
hold on
surf(X,Y,zmax*ones(size(X)))
surf(X,Y,zmin*ones(size(X)))
shading interp
if nargout>0
    varargout{1}=zmax;varargout{2}=zmin;
end
web -broswer http://www.ilovematlab.cn/forum-221-1.html