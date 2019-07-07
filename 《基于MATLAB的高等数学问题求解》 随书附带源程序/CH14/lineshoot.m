function [t,y] = lineshoot(f1,f2,tspan,x0f,varargin)
%LINESHOOT   ���Ա�ֵ����Ĵ�з����
% [T,Y] = LINESHOOT(F1,F2,TSPAN,X0F)  ��з������Ա�ֵ����F1�Ľ�
% [T,Y] = LINESHOOT(F1,F2,TSPAN,X0F,P1,P2,...)  ��з������Ա�ֵ����F1�Ľ⣬����
%                                               F1��F2�������Ӳ���P1,P2,...
%
% ���������
%     ---F1,F2��΢�ַ��̼����Ӧ��η��̵ĺ�������
%     ---TSPAN���������
%     ---X0F����ֵ����
%     ---P1,P2,...������F1��F2�ĸ��Ӳ���
% ���������
%     ---T�����صĽڵ�
%     ---Y����ֵ����Ľ�
%
% See also ode45

[~,y1] = ode45(f2,tspan,[1;0],varargin);  % ���㺯��y_1(t)
[~,y2] = ode45(f2,tspan,[0;1],varargin);  % ���㺯��y_2(t)
[~,yp] = ode45(f1,tspan,[0;0],varargin);  % ���㺯��y_p(t)
m = (x0f(2)-x0f(1)*y1(end,1)-yp(end,1))/y2(end,1);  % �����m
[t,y] = ode45(f1,tspan,[x0f(1);m],varargin);  % ���ԭ΢�ַ��̵Ľ�
web -broswer http://www.ilovematlab.cn/forum-221-1.html