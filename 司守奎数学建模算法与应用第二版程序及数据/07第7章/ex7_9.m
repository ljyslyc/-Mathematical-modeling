clc, clear,alpha=0.1;
edges=[-inf 20:10:100 inf]; %ԭʼ��������ı߽�
x=[25:10:95]; %ԭʼ������������� 
num=[5 15 30 51 60 23 10 6];
pd=@(x)normcdf(x,60,15); %������̬�ֲ��ķֲ�����
[h,p,st]=chi2gof(x,'cdf',pd,'Edges',edges,'Frequency',num)
pi=st.E/sum(st.O) %������еĵ�3������
col4=st.E %��ʾ���еĵ�4������
col5=st.O.^2./col4  %������еĵ�5������
sumcol5=sum(col5)  %������е�5�����ݵĺ�
k2=chi2inv(1-alpha,st.df) %���ٽ�ֵ

