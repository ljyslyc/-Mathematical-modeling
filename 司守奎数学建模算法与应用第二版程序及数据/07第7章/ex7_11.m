clc, clear
a=textread('ex7_5.txt'); a=nonzeros(a); %�������ݣ�ȥ��������㲢չ����������
xbar=mean(a), s=std(a), s2=var(a) %���ֵ, ��׼��ͷ���
[yn,xn]=cdfcalc(a);  %���㾭��ֲ�����ֵ
yn(end)=[];  %yn��Ԫ�ظ�����xn����һ����ɾ�����һ��ֵ
y=normcdf(xn,xbar,s); %�������۷ֲ�����ֵ
Dn=max(abs(yn-y))  %����ͳ������ֵ
LJ=1.36/sqrt(length(a)) %����ܾ�����ٽ�ֵ
%����ֱ�ӵ���Matlab��������������KS����
pd=makedist('Normal','mu',xbar,'sigma',s)
[h,p,st]=kstest(a,'CDF',pd) %ֱ�ӵ��ù�������������KS����
