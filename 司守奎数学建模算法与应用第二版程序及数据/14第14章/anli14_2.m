clc, clear
a=load('zhaopin.txt');   %��ԭʼ���ݱ����ڴ��ı��ļ�zhaopin.txt�У����Ұ�A��B��C��D�ֱ��滻����Ӧ����ֵ
b=zscore(a); %���ݱ�׼��
E=[1 4 2 8 2; 1/4 1 1/2 2 1/2; 1/2 2 1 4 1; 1/8 1/2 1/4 1 1/4; 1/2 2 1 4 1];
[vec, val]=eigs(E,1) %��ģ��������ֵ����Ӧ����������
w=vec/sum(vec)  %���һ��������������Ȩ��
w=repmat(w',16,1); %����Ϊ�����ݾ�����ͬ��ά��
c=b.*w    %�����Ȩ����
cstar=max(c)    %���������
c0=min(c)       %�������
for i=1:16
    sstar(i)=norm(c(i,:)-cstar);   %���������ľ���
    s0(i)=norm(c(i,:)-c0);       %�󵽸�����ľ���
end
f=s0./(sstar+s0);
xlswrite('book3.xls',[sstar' s0' f'])  %�Ѽ�����д��Excel�ļ��У����ڽ�������
[sc,ind]=sort(f,'descend')       %��������
