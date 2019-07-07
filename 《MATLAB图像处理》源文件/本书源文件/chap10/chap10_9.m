%[10.9]
close all; clear all; clc;						%�ر�����ͼ�δ��ڣ���������ռ����б��������������
J=imread('eye.bmp');  				%װ��ͼ���� Yucebianma��������Ԥ����룬��Yucejiema����
X=double(J);
Y=Yucebianma(X);
XX=Yucejiema(Y);
e=double(X)-double(XX);[m,n]=size(e);
erm=sqrt(sum(e(:).^2)/(m*n));	
set(0,'defaultFigurePosition',[100,100,1000,500]);%�޸�ͼ��ͼ��λ�õ�Ĭ������
set(0,'defaultFigureColor',[1 1 1])
figure,
subplot(121);imshow(J);
subplot(122),imshow(mat2gray(255-Y));	 %Ϊ������ʾ����Ԥ�����ͼȡ����������ʾ
figure;	
[h,x]=hist(X(:));%��ʾԭͼֱ��ͼ
subplot(121);bar(x,h,'k');
[h,x]=hist(Y(:));
subplot(122);bar(x,h,'k');
%Yucebianma ������һάԤ�����ѹ��ͼ��x,fΪԤ��ϵ�������fȥĬ��ֵ����Ĭ��f=1,����ǰֵԤ��
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
%Yucejiema�ǽ���������������õ���ͬһ��Ԥ����
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
