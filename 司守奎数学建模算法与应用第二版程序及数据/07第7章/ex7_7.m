clc, clear, n=100;
f=0:7; num=[36  40  19  2   0   2   1   0]; 
lamda=dot(f,num)/100
pi=poisspdf(f,lamda) 
[h,p,st]=chi2gof(f,'ctrs',f,'frequency',num,'expected',n*pi,'nparams',1) %���ù�����
col3=st.E/sum(st.O) %������еĵ�3������
col4=st.E %��ʾ���еĵ�4������
col5=st.O.^2./col4  %������еĵ�5������
sumcol5=sum(col5)  %������е�5�����ݵĺ�
k2=chi2inv(0.95,st.df)  %���ٽ�ֵ��st.dfΪ���ɶ�
