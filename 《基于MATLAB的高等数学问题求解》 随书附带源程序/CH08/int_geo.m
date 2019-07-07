function [I,Interval,s]=int_geo(fun,a,b,N)
%INT_GEO  ���ݶ����ֵļ��������󶨻���
% I=INT_GEO(FUN,A,B)  ���ö����ֵļ���������㺯��FUN�Ļ���ֵ�����������޷ֱ�ΪB��A
% I=INT_GEO(FUN,A,B,N)  ���ö����ֵļ���������㶨���֣�����ȷ���ΪN
% [I,INTERVAL]=INT_GEO(...)  ���ö����ֵļ���������㶨���֣������غ�����㸺����
% [I,INTERVAL,S]=INT_GEO(...)  ���ö����ֵļ���������㶨���֣������غ�����㸺����
%                                    �Լ���Ӧ�����ϵĻ���ֵ
%
% ���������
%     ---FUN��������MATLAB����������Ϊ��������������������M�ļ�
%     ---A,B���������޺�����
%     ---N����������㸺����ȷ���
% ���������
%     ---I������ֵ
%     ---INTERVAL��������㸺����
%     ---S���������ϵĻ���ֵ
%
% See also RootInterval, bisect, diff

if nargin<4
    N=1000;
end
r=RootInterval(fun,a,b);
if ~isempty(r)
    n=size(r,1);
    x=ones(1,n+2);
    x(1)=a; x(end)=b;
    for k=1:n
        x(k+1)=bisect(fun,r(k,1),r(k,2));
    end
    x=unique(x);
    L=length(x);
    Interval=zeros(2,L-1);
    for kk=1:L-1
        Interval(:,kk)=x(kk:kk+1);
    end
else
    Interval=[a;b];
end
h=diff(Interval)/N;
M=mean(Interval);
fM=feval(fun,M);
fM(fM>0)=1;
fM(fM<0)=-1;
s=zeros(1,size(Interval,2));
for k=1:size(Interval,2)
    xx=Interval(1,k)+h(k)*(0:N);
    fx=abs(feval(fun,xx));
    s(k)=sum(fx)*h(k);
end
I=sum(s.*fM);
web -broswer http://www.ilovematlab.cn/forum-221-1.html