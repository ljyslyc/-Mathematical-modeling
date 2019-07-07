function [x,y]=Explicit_Euler(odefun,xspan,y0,h,varargin)
%EXPLICIT_EULER   ŷ��������ֵ�������ֵ��
% [X,Y]=EXPLICIT_EULER(ODEFUN,XSPAN,Y0,H)  ŷ������΢�ַ���ODEFUN����ֵ��
% [X,Y]=EXPLICIT_EULER(ODEFUN,XSPAN,Y0,H,P1,P2,...)  ŷ������΢�ַ���ODEFUN
%                                      ����ֵ�⣬ODEFUN���и��Ӳ���P1,P2,...
%
% ���������
%     ---ODEFUN��΢�ַ��̵ĺ�������
%     ---XSPAN���������[x0,xn]
%     ---Y0����ʼ����
%     ---H����������
%     ---P1,P2,...��ODEFUN�����ĸ��Ӳ���
% ���������
%     ---X�����صĽڵ㣬��X=XSPAN(1):H:XSPAN(2)
%     ---Y��΢�ַ��̵���ֵ��
%
% See also ode*

x=xspan(1):h:xspan(2);
N=length(x);
y=zeros(1,N);
y(1)=y0;
for k=1:N-1
    y(k+1)=y(k)+h*feval(odefun,x(k),y(k),varargin{:});
end
x=x';y=y';
web -broswer http://www.ilovematlab.cn/forum-221-1.html