clc, clear
a=textread('ex7_5.txt'); a=nonzeros(a); %�������ݣ�ȥ��������㲢չ����������
xbar=mean(a), s=std(a) %���ֵ�ͱ�׼��
pd=ProbDistUnivParam('normal',[xbar s]) %������̬�ֲ�
qqplot(a,pd)  %Matlab������ֱ�ӻ�Q-Qͼ
%���治���ù����仭Q-Qͼ
sa=sort(a); %��a���մ�С��������
n=length(a); pi=([1:n]-1/2)/n;
yi=norminv(pi,xbar,s)' %�����Ӧ��yiֵ
hold on, plot(yi,sa,'.') %������㻭Q-Qͼ
