%Yucejiema�ǽ���������������õ���ͬһ��Ԥ����������Yucejiema.m�ļ�
function x=Yucejiema(y,f)
error(nargchk(1,2,nargin));
if nargin<2
  f=1;
end
f=f(end:-1:1);
[m,n]=size(y);
order=length(f);
f=repmat(f,m,1);
x=zeros(m,n+order);
for j=1:n
  jj=j+order;
  x(:,jj)=y(:,j)+round(sum(f(:,order:-1:1).*x(:,(jj-1):-1:(jj-order)),2));
end
x=x(:,order+1:end);
