%Yucebianma ������һάԤ�����ѹ��ͼ��x,fΪԤ��ϵ��������Yucebianma.m�ļ�
function y=Yucebianma(x,f)
error(nargchk(1,2,nargin))
if nargin<2
  f=1;
end
x=double(x);
[m,n]=size(x); 
p=zeros(m,n);  					 %���Ԥ��ֵ
xs=x;
zc=zeros(m,1);
for j=1:length(f)
    xs=[zc xs(:,1:end-1)];
    p=p+f(j)*xs;
end
y=x-round(p);
