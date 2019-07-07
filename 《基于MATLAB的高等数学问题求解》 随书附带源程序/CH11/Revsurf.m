function varargout=Revsurf(x,fun,type)
%REVSURF   ������ת��
% REVSURF(X,FUN)  ��������FUN��Z����ת���õ���ת�壬����FUN�ǹ���Z�ĺ���
% REVSURF(X,FUN,TYPE)  ����FUN��Z����ת���õ���ת�壬����FUN��TYPEָ���Ա���
% H=REVSURF(...)  ������ת�岢������ת��ͼ�ξ��
% [XX,YY,ZZ]=REVSURF(...)  ������ת����������
%
% ���������
%     ---X���������Ա�������
%     ---FUN���������ߵĺ���
%     ---TYPE��ָ���������Ա�����TYPE����������ȡֵ��
%              1.'cylinder'��1��FUN�ǹ���Z�ĺ���
%              2.'revsurf'��2��FUN�ǹ���X��Y�ĺ���
% ���������
%     ---H����ת��ͼ�ξ��
%     ---XX,YY,ZZ����ת����������
%
% See also cylinder

if nargin==2
    type='cylinder';
end
switch lower(type)
    case {1,'cylinder'}
        xL=min(x(:)); xR=max(x(:));
        [xx,yy,zz]=cylinder(fun(x),40);
        zz=xL+(xR-xL)*zz;
    case {2,'revsurf'}
        [theta,rho]=meshgrid(linspace(0,2*pi),x);
        [xx,yy]=pol2cart(theta,rho);
        R=sqrt(xx.^2+yy.^2);
        zz=fun(R);
    otherwise
        error('Illegal options.')
end
if nargout==0
    surf(xx,yy,zz)
elseif nargout==1
    varargout{1}=surf(xx,yy,zz);
elseif nargout==3
    varargout{1}=xx; varargout{2}=yy; varargout{3}=zz;
else
    error('The number of output arguments is Illegal.')
end
web -broswer http://www.ilovematlab.cn/forum-221-1.html