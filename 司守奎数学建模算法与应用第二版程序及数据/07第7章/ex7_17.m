clc, clear
x0=[342  500  187];
theta=(x0(2)+2*x0(3))/sum(x0)/2  %���������Ȼ����
fb=[(1-theta)^2,2*theta*(1-theta),theta^2]  %����ֲ���
cf=cumsum(fb)  %���ۼƷֲ�
a=rand(1029,1000);  %ÿһ���������Ӧһ��bootstrap����
jx1=(a<=cf(1));  %1��ӦM����
jx2=(a>cf(1) & a<=cf(2)); %1��ӦMN���� 
jx3=(a>=cf(2)); %1��ӦN����
x1=sum(jx1); x2=sum(jx2); x3=sum(jx3);  
theta2=(x2+2*x3)/1029/2;  %����ͳ����theta��ֵ
stheta=sort(theta2); %��ͳ�������մ�С��������
qj=[stheta(50), stheta(950)]  %������������ȡֵ
