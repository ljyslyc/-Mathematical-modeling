clc, clear
x0=[506  508  499  503  504  510  497  512
514  505  493  496  506  502  509  496]; x0=x0(:);
alpha=0.05;
mu=mean(x0), sig=std(x0), n=length(x0);
t=[mu-sig/sqrt(n)*tinv(1-alpha/2,n-1),mu+sig/sqrt(n)*tinv(1-alpha/2,n-1)]
%��������ttest�ķ���ֵci��ֱ�Ӹ����������������
[h,p,ci]=ttest(x0,mu,0.05)  %ͨ���������Ҳ�������������
