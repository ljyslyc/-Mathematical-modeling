clc, clear
elps=randn(10000,1); x(1:2)=0;
for i=3:10000
    x(i)=-0.6*x(i-1)-0.2*x(i-2)+elps(i); %����ģ������
end
xlswrite('data1.xls',x(end-9:end)) %��x�ĺ�10�����ݱ��浽Excel�ļ���
dlmwrite('mydata.txt',x)  %��������8.13��GARCHģ��ʹ��ͬ��������
x=x'; m=ar(x,2)   %���в�������
xhat=forecast(m,x,3) %����3��Ԥ��ֵ
