clc,clear
x=load('water.txt');  %��ԭʼ���ݰ��ձ��еĸ�ʽ����ڴ��ı��ļ�water.txt
x=x'; x=x(:);  %����ʱ����Ⱥ���򣬰����ݱ��������
s=12;  %����s=12
n=12;  %Ԥ�����ݵĸ���
m1=length(x);   %ԭʼ���ݵĸ���
for i=s+1:m1
    y(i-s)=x(i)-x(i-s); %�������ڲ�ֱ任
end
w=diff(y);   %���������ԵĲ������
m2=length(w); %�������ղ�ֺ����ݵĸ���
k=0; %��ʼ����̽ģ�͵ĸ���
for i=0:3
    for j=0:3
        if i==0 & j==0
            continue
        elseif i==0
            ToEstMd=arima('MALags',1:j,'Constant',0); %ָ��ģ�͵Ľṹ
        elseif j==0
            ToEstMd=arima('ARLags',1:i,'Constant',0); %ָ��ģ�͵Ľṹ
        else
            ToEstMd=arima('ARLags',1:i,'MALags',1:j,'Constant',0); %ָ��ģ�͵Ľṹ
        end
        k=k+1; R(k)=i; M(k)=j;
        [EstMd,EstParamCov,logL,info]=estimate(ToEstMd,w'); %ģ�����
        numParams = sum(any(EstParamCov)); %������ϲ����ĸ���
        %compute Akaike and Bayesian Information Criteria
        [aic(k),bic(k)]=aicbic(logL,numParams,m2);
    end
end
fprintf('R,M,AIC,BIC�Ķ�Ӧֵ����\n %f');  %��ʾ������
check=[R',M',aic',bic']
r=input('�������R��');m=input('�������M��');
ToEstMd=arima('ARLags',1:r,'MALags',1:m,'Constant',0); %ָ��ģ�͵Ľṹ
[EstMd,EstParamCov,logL,info]=estimate(ToEstMd,w'); %ģ�����
w_Forecast=forecast(EstMd,n,'Y0',w')  %����12��Ԥ��ֵ,ע����֪������������
yhat=y(end)+cumsum(w_Forecast)     %��һ�ײ�ֵĻ�ԭֵ
for j=1:n
    x(m1+j)=yhat(j)+x(m1+j-s); %��x��Ԥ��ֵ
end
xhat=x(m1+1:end)   %��ȡn��Ԥ��ֵ
