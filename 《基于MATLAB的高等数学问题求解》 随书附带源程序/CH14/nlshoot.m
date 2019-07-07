function [t,y] = nlshoot(f1,fv,tspan,x0f,tol,varargin)
%NLSHOOT   �����Ա�ֵ����Ĵ�з����
% [T,Y] = NLSHOOT(F1,FV,TSPAN,X0F,TOL)  ��з�������Ա�ֵ����F1�Ľ�
% [T,Y] = NLSHOOT(F1,FV,TSPAN,X0F,TOL,P1,P2,...)  ��з�������Ա�ֵ����F1�Ľ⣬
%                                              ����F1��FV�������Ӳ���P1,P2,...
%
% ���������
%     ---F1,FV��΢�ַ�����ǰ����ܵĹ��ڱ���v1,v2,v3,v4��΢�ַ��̵ĺ�������
%     ---TSPAN���������
%     ---X0F�������ı�ֵ����
%     ---TOL������Ҫ�����ڿ��Ʋ���m�����
%     ---P1,P2,...������F1��FV�ĸ��Ӳ���
% ���������
%     ---T�����صĽڵ�
%     ---Y����ֵ����Ľ�
%
% See also ode45, lineshoot

m0=1;  % m�ĳ�ֵ
err=1;
while abs(err)>tol;
    [~,v] = ode45(fv,tspan,[x0f(1);m0;0;1],varargin);  % �������ʽ
    m=m0-(v(end,1)-x0f(2))/v(end,3);  % ����m����ֵ
    err=m-m0;
    m0=m;
end
[t,y] = ode45(f1,tspan,[x0f(1);m],varargin);  % ���õõ��ĳ�ֵ��ⷽ��
web -broswer http://www.ilovematlab.cn/forum-221-1.html