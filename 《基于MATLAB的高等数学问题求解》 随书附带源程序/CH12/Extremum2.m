function [xmax,fmax,xmin,fmin]=Extremum2(fun)
%EXTREMUM2   ���Ԫ�����ļ�ֵ�뼫ֵ��
% [XMAX,FMAX,XMIN,FMIN]=EXTREMUM2(FUN)  ���Ԫ����FUN�ļ�ֵ�뼫ֵ��
%
% ���������
%     ---FUN����Ԫ�������ű��ʽ
% ���������
%     ---XMAX,XMIN������ֵ�ͼ�Сֵ��
%     ---FMAX,FMIN������ֵ�ͼ�Сֵ
%
% See also diff

if ~isa(fun,'sym')
    error('FUN must be a Symbolic function.')
end
dfx=diff(fun,'x');
dfy=diff(fun,'y');
[x0,y0]=solve(dfx,dfy,'x','y');
xmax=[];xmin=[];
for k=1:length(x0)
    A=subs(diff(dfx,'x'),{'x','y'},{x0(k),y0(k)});
    B=subs(diff(dfx,'y'),{'x','y'},{x0(k),y0(k)});
    C=subs(diff(dfy,'y'),{'x','y'},{x0(k),y0(k)});
    if double(A*C-B^2)>0
        if double(A)<0
            xmax=[xmax;[x0(k),y0(k)]];
        else
            xmin=[xmin;[x0(k),y0(k)]];
        end
    end
end
if ~isempty(xmax)
    fmax=subs(fun,{'x','y'},{xmax(:,1),xmax(:,2)});
else
    fmax=[];
end
if ~isempty(xmin)
    fmin=subs(fun,{'x','y'},{xmin(:,1),xmin(:,2)});
else
    fmin=[];
end
web -broswer http://www.ilovematlab.cn/forum-221-1.html