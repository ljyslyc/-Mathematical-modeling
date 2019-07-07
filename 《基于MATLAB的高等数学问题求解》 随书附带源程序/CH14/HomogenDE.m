function varargout=HomogenDE(fun,coef,t)
%HOMOGENDE   ��λ�ɻ�Ϊ��η��̵����
% L=HOMOGENDE(FUN,COEF)  ��΢�ַ���dY/dX=FUN((a*x+b*y+c)/(a1*x+b1*y+c1))��ͨ��
% L=HOMOGENDE(FUN,COEF,T)  ��ɻ�Ϊ��η��̵�ͨ�⣬��ָ��FUN���Ա���ΪT
% [L,S]=L=HOMOGENDE(...)  ��ɻ�Ϊ��η��̵�ͨ�Ⲣ���������ַ�����ʽ
%
% ���������
%     ---FUN������(a*x+b*y+c)/(a1*x+b1*y+c1)�ĺ���
%     ---COEF��ϵ������[a,b,c;a1,b1,c1]
%     ---T������FUN���Ա���
% ���������
%     ---L��΢�ַ��̵�ͨ��
%     ---S��΢�ַ��̽���ַ�����ʾ
%
% See also SeparableVarsDE

if nargin==2
    t=symvar(fun);
end
if length(t)>1
    error('���ű�����������.')
end
syms x y
D=det(coef(:,1:2));
if D==0
    v=sym('v','real');
    L=coef(2,1)/coef(1,1);
    fun=subs(fun,t,(v+coef(1,3))/(L*v+coef(2,3)));
    I=SeparableVarsDE(sym(coef(1,2)),1/(fun+coef(1,1)/coef(1,2)),x,v);
    yy=subs(I,v,coef(1,1)*x+coef(1,2)*y);
else
    u=sym('u','real');
    X=sym('X','real');
    Y=sym('Y','real');
    x0=-coef(:,1:2)\coef(:,3);
    fun=subs(fun,t,(coef(1,1)+coef(1,2)*u)/(coef(2,1)+coef(2,2)*u));
    I=SeparableVarsDE(1/X,1/(fun-u),X,u);
    I=subs(I,u,Y/X);
    yy=subs(I,{X,Y},{x-x0(1),y-x0(2)});
end
varargout{1}=yy;
if nargout==2
    varargout{2}=['Solution:',char(yy)];
end
web -broswer http://www.ilovematlab.cn/forum-221-1.html