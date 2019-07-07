function I=dbldefinition(fun,D,m,n)
%DBLDEFINITION   ���ݶ��ػ��ֵĶ��������ػ���
% I=DBLDEFINITION(FUN,D,M)  ���㺯��FUN������D�ϵĶ��ػ��֣�D��ΪM*M����
% I=DBLDEFINITION(FUN,D,M,N)  ���㺯��FUN������D�ϵĶ��ػ��֣�D��ΪM*N����
%
% ���������
%     ---FUN����Ԫ������MATLAB��������������������������������
%     ---D����������
%     ---M,N����������Ļ�����
% ���������
%     ---I�����ػ���ֵ
%
% See also sum, diff

if nargin<4
    n=m;
end
a=min(D(1,:));
b=max(D(1,:));
c=min(D(2,:));
d=max(D(2,:));
x=linspace(a,b,m);
y=linspace(c,d,n);
[X,Y]=meshgrid(x,y);
in=inpolygon(X(:),Y(:),D(1,:),D(2,:));
f=fun(X(in),Y(in));
I=sum(f*diff(x(1:2))*diff(y(1:2)));
web -broswer http://www.ilovematlab.cn/forum-221-1.html