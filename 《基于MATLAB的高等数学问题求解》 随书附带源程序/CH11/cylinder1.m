function varargout=cylinder1(x,y,N)
%CYLINDER1   ��������
% CYLINDER1  ���Ƶ���Բ��Բ����ԭ�㣬�뾶Ϊ1���߶�Ϊ1��Բ��
% CYLINDER1(X,Y)  ������X��Y���ɵ�����Ϊĸ�ߵĸ߶�Ϊ1������
% CYLINDER1(X,Y,N)  ������X��Y���ɵ�����Ϊĸ�ߵ����棬������߶ȷ�ΪN�ȷ�
% H=CYLINDER1(...)  �������沢��������
% [XX,YY,ZZ]=CYLINDER1(...)  ����������������
%
% ���������
%     ---X,Y��ĸ�ߵ���������
%     ---N������߶ȵĵȷ���
% ���������
%     ---H������ľ��
%     ---XX,YY,ZZ��������������
%
% See also cylinder

if nargin<3
    N=2;
end
t=linspace(0,2*pi);
if nargin<1
    x=cos(t);y=sin(t);
end   
if length(x)~=length(y)
    error('��������ά����ƥ��.')
end
x=x(:); y=y(:);
X=repmat(x,1,N);
Y=repmat(y,1,N);
Z=repmat(linspace(0,1,N),length(x),1);
if nargout==0
    surf(X,Y,Z)
elseif nargout==1
    h=surf(X,Y,Z);
    varargout{1}=h;
else
    varargout{1}=X; varargout{2}=Y; varargout{3}=Z;
end
web -broswer http://www.ilovematlab.cn/forum-221-1.html