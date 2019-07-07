function varargout=quadric(varargin)
%QUADRIC   ���ƶ�������
% QUADRIC('elliptic',XC,YC,ZC,A,B,N)  ������Բ׶��
% QUADRIC('ellipsoid',XC,YC,ZC,A,B,C,N)  ����������
% QUADRIC('hyperboloidofonesheet',XC,YC,ZC,A,B,C,N)  ���Ƶ�Ҷ˫����
% QUADRIC('hyperboloidoftwosheets',XC,YC,ZC,A,B,C,N)  ����˫Ҷ˫����
% QUADRIC('ellipticparaboloid',XC,YC,ZC,A,B,N)  ������Բ������
% QUADRIC('hyperbolicparaboloid',A,B,N)  ����˫��������
% H=QUADRIC(...)  ���ƶ������沢��������
% [X,Y,Z]=QUADRIC(...)  ��������������������
%
% ���������
%     ---XC,YC,ZC�������������������
%     ---A,B,C����������Ĳ���
%     ---N��ָ����������
%     ---TYPE��ָ�������������ͣ�������6��ȡֵ
% ���������
%     ---H����������ľ��
%     ---X,Y,Z�������������������
%
% See also cylinder, ellipsoid

args=varargin;
type=args{1};
switch lower(type)
    case {1,'elliptic','��Բ׶��'}
        [xc,yc,zc,a,b,n]=deal(args{2:end});
        z=linspace(-a,a);
        [X,Y,Z]=cylinder(a*z,n);
        X=X+xc;
        Y=b/a*Y+yc;
        Z=-a+2*a*Z+zc;
    case {2,'ellipsoid','������'}
        [xc,yc,zc,a,b,c,n]=deal(args{2:end});
        [X,Y,Z]=ellipsoid(xc,yc,zc,a,b,c,n);
    case {3,'hyperboloidofonesheet','��Ҷ˫����'}
        % �������̣�
        % x=a*sec(t)*cos(p)
        % y=b*sec(t)*sin(p)
        % z=c*tan(t)
        [xc,yc,zc,a,b,c,n]=deal(args{2:end});
        t=linspace(-pi/2.5,pi/2.5,n);
        p=linspace(-pi,pi,30);
        [T,P]=meshgrid(t,p);
        X=a*sec(T).*cos(P)+xc;
        Y=b*sec(T).*sin(P)+yc;
        Z=c*tan(T)+zc;
    case {4,'hyperboloidoftwosheets','˫Ҷ˫����'}
        [xc,yc,zc,a,b,c,n]=deal(args{2:end});
        t=linspace(-pi/2.5,pi/2.5,n);
        p=linspace(-pi,pi,30);
        [T,P]=meshgrid(t,p);
        X=a*sec(T)+xc;
        Y=b*tan(T).*cos(P)+yc;
        Z=c*tan(T).*sin(P)+zc;
    case {5,'ellipticparaboloid','��Բ������'}
        [xc,yc,zc,a,b,n]=deal(args{2:end});
        z=linspace(0,abs(a));
        [X,Y,Z]=cylinder(abs(a)*sqrt(z),n);
        X=X+xc;
        Y=b/a*Y+yc;
        Z=Z+zc;
    case {6,'hyperbolicparaboloid','˫��������'}
        [a,b,n]=deal(args{2:end});
        x=linspace(-a^2, a^2,n);
        y=linspace(-b^2, b^2,n);
        [X,Y]=meshgrid(x,y);
        Z=X.^2/a^2-Y.^2/b^2;
end
if nargout==0
    surf(X,Y,Z)
elseif nargout==1
    h=surf(X,Y,Z);
    varargout{1}=h;
elseif nargout==3
    varargout{1}=X; varargout{2}=Y; varargout{3}=Z;
else
    error('The Number of output arguments is wrong.')
end
web -broswer http://www.ilovematlab.cn/forum-221-1.html