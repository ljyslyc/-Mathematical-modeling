function PI=CalculatePI(n)
%CALCULATEPI   Բ����PI�ļ����㷨
% PI=CALCULATEPI(N)  �����ݼ�������Բ���ʵ�ֵ
%
% ���������
%     ---N��������ȡ������
% ���������
%     ---PI��Բ���ʵĽ���ֵ
%
% See also pi

if nargin==0
    n=1000;
end
PI=0;
for k=1:n
    a=(-1)^(k-1)/(2*k-1);
    PI=PI+a;
end
PI=4*PI;
web -broswer http://www.ilovematlab.cn/forum-221-1.html