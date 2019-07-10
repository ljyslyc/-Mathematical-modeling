clc,clear
VarMd=garch('Constant',0.01,'GARCH',0.2,'ARCH',0.3); %ָ��ģ�͵Ľṹ
Md=arima('Constant',0,'AR',0.8,'MA',0.4, 'Variance',VarMd); %ָ��ģ�͵Ľṹ
[y,e,v]=simulate(Md,10000); %����ָ���ṹģ�͵�10000��ģ������
ToEstVarMd=garch(1,1);
ToEstMd=arima('ARLags',1,'MALags',1,'Constant',0,'Variance',ToEstVarMd);
[EstMd,EstParamCov,logL,info]=estimate(ToEstMd,y)  %ģ�����
res=infer(EstMd,y); %����в�
h=lbqtest(res) %����ģ�ͼ���
yhat=forecast(EstMd,3,'Y0',y) %Ԥ��δ����3��ֵ
