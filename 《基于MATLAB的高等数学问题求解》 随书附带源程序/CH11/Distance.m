function D=Distance(A,B)
%DISTANCE   ��������֮��ľ���
% DISTANCE(A)  ����ԭ�㵽��A��������ͼ���עΪԭ�㵽A�ľ���
% DISTANCE(A,B)   ���Ƶ�B����A��������ͼ���עΪA��B��ľ���
% D=DISTANCE(...)  ����ԭ�㵽A��A��B֮��ľ���
%
% ���������
%     ---A���յ�
%     ---B�����
% ���������
%     ---D��A��B��ľ���
%
% See also norm, sqrt

if nargin==1
    B=zeros(size(A));
end
[m,n]=size(A);
if ~isequal([m,n],size(B)) || m~=1
    error('��������ʾ��ʽ����.')
end
C=A-B;
L=0;
for k=1:n
    L=L+C(k)^2;
end
L=sqrt(L);
if isnumeric([A,B]) && (n==2 || n==3) && nargout==0
    drawvec(C,B)
    title(['|AB|=',num2str(L)])
elseif nargout==1
    D=L;
end
web -broswer http://www.ilovematlab.cn/forum-221-1.html