function surface_para(funx,funy,funz,varargin)
%SURFACE_PARA   �����Բ������̱�ʾ��������Բ������̱�ʾ��������z����ת���õ���ת����
% SURFACE_PARA(FUNX,FUNY,FUNZ,T)  ���Ʋ�������ȷ����������z����ת����ת����
% SURFACE_PARA(FUNX,FUNY,FUNZ,U,V)  ���Ʋ�������ȷ��������
%
% ���������
%     ---FUNX,FUNY,FUNZ���������̣����������߻�����
%     ---T�����߲��������Ա���
%     ---U,V������������̵��Ա���
%
% See also surf

s=unique([symvar(funx),symvar(funy),symvar(funz)]);
if length(s)==1
    theta=linspace(0,2*pi);
    t=varargin{1};
    [T,Th]=meshgrid(t,theta);
    X=subs(sqrt(funx^2+funy^2),s,T).*cos(Th);
    Y=subs(sqrt(funx^2+funy^2),s,T).*sin(Th);
    Z=subs(funz,s,T);
elseif length(s)==2
    [u,v]=deal(varargin{:});
    [U,V]=meshgrid(u,v);
    X=subs(funx,num2cell(s),{U,V});
    Y=subs(funy,num2cell(s),{U,V});
    Z=subs(funz,num2cell(s),{U,V});
else
    error('�������̵Ĳ�����������.')
end
surf(X,Y,Z)
web -broswer http://www.ilovematlab.cn/forum-221-1.html