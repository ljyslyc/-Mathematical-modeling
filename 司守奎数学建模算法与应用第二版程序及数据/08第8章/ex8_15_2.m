clc,clear
a=textread('hua.txt');  %��ԭʼ���ݰ���ԭ�������и�ʽ����ڴ��ı��ļ�hua.txt
a=nonzeros(a'); %����ԭ�����ݵ�˳��ȥ����Ԫ��
da=diff(a);      %����1�ײ��
ToEstMd=arima('MALags',1); %ָ��ģ�͵Ľṹ
[EstMd,EstParamCov,logL,info]=estimate(ToEstMd,da); %ģ�����
dx_Forecast=forecast(EstMd,10,'Y0',da)  %����10��Ԥ��ֵ
x_Forecast=a(end)+cumsum(dx_Forecast)   %����ԭʼ���ݵ�10��Ԥ��ֵ
