clc, clear
randn('seed',sum(100*clock));  %��ʼ�������������
a=randn(6,1); %���ɷ��ӱ�׼��̬�ֲ���α�����
b=[today:today+5]'  %�ӽ��쵽����5��
fts=fints(b,a)  %����fints��ʽ����
fts(3)=NaN;  %����3�����ݱ�ΪȱʧֵNaN
newdata=fillts(fts,'linear') %�����Բ�ֵ�ʱ�������е�ȱʧ����
data=fts2mat(newdata) %ʱ����������ת����ͨ����
