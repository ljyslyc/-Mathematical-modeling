function [yc0]=gm12(x0,number)
n=length(x0);
global N
for i=1:n
x1(i)=sum(x0(1:i));
end

% for k=2:n % ���ھ����� z
% z(k)=0.5*x1(k)+0.5*x1(k-1);
% end
z=(x1(1:n-1)+x1(2:n))./2;

% for i=1:n-1
% b(i,1)=-z(i+1);
% y(i)=x0(i+1);
% end
b(:,1)=-z';
b(:,2)=1;
y=x0(2:n);

y=y'; % ת��Ϊ������
au=b\y % ���������������a u

yc1(1)=x0(1);
c=x0(1)-au(2)/au(1);

for k=1:n+N-1
yc1(k+1)=c*exp( -au(1)*k)+au(2)/au(1);
end

yc0(1)=x0(1);
% for k=1:n+N
% yc0(k+1)=yc1(k+1)-yc1(k);
% end
yc0(2:n+N)=yc1(2:n+N)-yc1(1:n+N-1);
% disp(uint16(yc0(2:1:n+1)));

% for k=1:n
% e0(k)=x0(k)-yc0(k);
% end
e0=x0(1:n)-yc0(1:n);
e02=100*(x0(1:n)-yc0(1:n))./x0(1:n);
max1=max(abs(e0));
r=1;
for k=2:n
r=r+0.5*max1/(abs(e0(k))+0.5*max1);
end
r=r/n; % r ��ʾ������

pe=mean(e0);
pe2=mean(e02);

% px0=sum(x0)/n;
% z=x0-px0;
% S1=sqrt(sum(z.^2)/n);
% z=e0-pe;
% S2=sqrt(sum(z.^2)/n);
% C=S2/S1
c= std(e0)./std(x0)