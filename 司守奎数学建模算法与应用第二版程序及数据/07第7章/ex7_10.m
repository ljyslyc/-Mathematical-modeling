clc, clear, alpha=0.1;
a=textread('ex7_5.txt'); a=nonzeros(a); %�������ݣ�ȥ��������㲢չ����������
xbar=mean(a), s=std(a) %���ֵ�ͱ�׼��
mm=minmax(a')  %��۲�ֵ�����ֵ����Сֵ
pd=@(x)normcdf(x,xbar,s); %������̬�ֲ�
[h,p,st]=chi2gof(a,'cdf',pd,'NParams',2)  %���ù�����ļ����������
pi=st.E/length(a) %�������
col4=st.E %��ʾ���еĵ�4������
tj=st.O.^2./st.E, stj=sum(tj) %������е����һ�м���
k2=chi2inv(1-alpha,st.df) %���ٽ�ֵ
