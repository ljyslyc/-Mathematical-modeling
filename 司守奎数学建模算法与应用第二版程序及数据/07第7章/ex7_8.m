clc, clear
edges=[0:100:300 inf]; bins=[50 150 250 inf]; %����ԭʼ��������ı߽������
num=[121 78 43 58]; %��֪�۲�Ƶ��
pd=makedist('exp',200)  %����ָ���ֲ�
[h,p,st]=chi2gof(bins,'Edges',edges,'cdf',pd,'Frequency',num) 
pi=st.E/sum(st.O) %������еĵ�3������
col4=st.E %��ʾ���еĵ�4������
col5=st.O.^2./col4  %������еĵ�5������
sumcol5=sum(col5)  %������е�5�����ݵĺ�
k2=chi2inv(0.95,st.df)  %���ٽ�ֵ��st.dfΪ���ɶ�
