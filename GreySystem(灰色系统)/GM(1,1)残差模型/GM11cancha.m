function [yc0]=gm11cancha(x0)
n=max(size(x0));    %�����С..
[yc0]=gm11(x0);%����GM(1,1)ģ�ͽ���Ԥ��
wucha=x0-yc0(1:n);%����в�ֵ
i=n;
%�����ͬ�ŵĲв����Ŀ.
while(wucha(i)*wucha(i-1)>0 & i>=2)
    i=i-1;
end
start=i;
start
length=n-i+1;
new=wucha(start:n);

if length>=4
    pwucha=gm12(new);
    n=max(size(x0));
    yc0=gm12(x0);%�Կɽ�ģ�в����GM(1,1)ģ�ͶԲв�ֵ����Ԥ��
yc0(start:n+N)=yc0(start:n+N)+pwucha %yc0Ϊ�в�غ��Ԥ������
clear wucha;
wucha=yc0(1:n)-x0;
wucha=wucha./x0;    %������
wucha=abs(wucha)*100;
rel=sum(wucha)/n;
end