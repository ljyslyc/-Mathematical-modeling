function r=CrossPoint(varargin)
%CROSSPOINT   ȷ��ͼ�����ڵ�ˮƽ����
% R=CROSSPOINT(F1,F2)  ��������F1��F2�Ľ�������꣬�����Ա���Ϊ�����Ա���
% R=CROSSPOINT(F1,F2,X)  ��������F1��F2�Ľ�������꣬�����Ա���ΪX
% 
% ���������
%     ---F1,F2�����ߵĺ�������
% ���������
%     ---R���������������
%
% See also solve

[f1,f2]=deal(varargin{1:2});
s=unique([symvar(f1),symvar(f2)]);
if nargin==2 && length(s)==1
    x=s;
else
    x=varargin{3};
end
x0=solve(f1-f2,x);
N=length(x0);
r=zeros(N-1,2);
for k=1:N-1
    r(k,:)=[x0(k),x0(k+1)];
end
web -broswer http://www.ilovematlab.cn/forum-221-1.html