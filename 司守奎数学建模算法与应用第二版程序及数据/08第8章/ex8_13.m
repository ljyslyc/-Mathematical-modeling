clc, clear
ToEstVarMd=garch(1,1);
%ToEstMd=arima('ARLags',1:2,'Variance',ToEstVarMd); %AR�Ľ״�ȡ2���޷�ͨ��
ToEstMd=arima('ARLags',1,'Variance',ToEstVarMd);
y=load('mydata.txt');
[EstMd,EstParamCov,logL,info]=estimate(ToEstMd,y')  %ģ�����,ע��yΪ������
yhat=forecast(EstMd,3,'Y0',y') %����3��Ԥ��ֵ
