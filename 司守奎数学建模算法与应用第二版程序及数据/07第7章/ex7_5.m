clc, clear
a=textread('ex7_5.txt'); a=nonzeros(a); %�������ݣ���ȥ���������չ��������
[ycdf,xcdf,n]=cdfcalc(a) %���㾭��ֲ�������ȡֵ
cdfplot(a), title('') %������ֲ�������ͼ��
hold on, plot(xcdf,ycdf(2:end),'.') %�����»�����ֲ�������ȡֵ
xlswrite('ex7_5.xls',[xcdf,ycdf(2:end)])
