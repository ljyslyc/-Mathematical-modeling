clc, clear
x0=[2.5320,2.6470,2.6290,2.5840,2.6090,2.6010,2.5280,2.5630,2.6540,2.6190];
n=length(x0);
me=quantile(x0,0.5)  %������λ��
[h,p,stat]=runstest(x0,me)  %�����γ̼���
x1=cumsum(x0);  %���ۼ�����
zk=(x1(1:end-1)+x1(2:end))/2  %���ۼ����еľ�ֵ����
B=[-zk', ones(size(zk'))]; yn=x0(2:end)';
ab=B\yn  %��ϲ���a,b
syms x(t)
x=dsolve(diff(x)+ab(1)*x==ab(2),x(0)==x0(1)); %��΢�ַ��̵ķ��Ž�
xx=vpa(x,6)  %��ʾС����ʽ�ķ��Ž�
yuce=subs(x,'t',[0:n+5]);  %���ۼ����е�Ԥ��ֵ
yuce=double(yuce); %�ѷ�����ת������ֵ����
yuce0=[x0(1),diff(yuce)]  %��ԭʼ���ݵ�Ԥ��ֵ
c=std(yuce0(1:n))/std(x0)  %�������ֵc
nyuce=yuce0(n+1:end)  %��ȡ6���µ�Ԥ��ֵ
nyb=[x0, nyuce];  %�����µ���������
nnyb=reshape(nyb,[4,4])
mu=mean(nnyb)  %�ֱ���4���������ľ�ֵ
jc=range(nnyb)  %�ֱ���4���������ļ���
xlswrite('hb.xls',[nnyb;mu;jc])  %������д��Excel�ļ��У���������ʹ��
b=rand(4,1000); %����4��1000�е����������
h=floor(b*length(nyb))+1; %�������ӳ��Ϊ���(ÿ�ж�Ӧbootstrap�������)
bb=repmat(nyb',1,1000); bb=bb(h); %�������н����ظ�����
mmu=mean(bb); mjc=range(bb); %����1000���������ľ�ֵ�ͼ���
smu=sort(mmu); sjc=sort(mjc); %�Ѿ�ֵ�ͼ���մ�С����Ĵ�������
alpha=0.0027; k1=floor(1000*alpha/2), k2=floor(1000*(1-alpha/2))
mqj=[smu(k1), smu(k2)]  %��ʾ��ֵ����������
jqj=[sjc(k1), sjc(k2)]  %��ʾ�������������
subplot(1,2,1), plot(mu,'*-'), hold on, plot([1,4],[mqj(1),mqj(1)])
plot([1,4],[mqj(2),mqj(2)]), ylabel('������ֵ')
subplot(1,2,2), plot(jc,'*-'), hold on, plot([1,4],[jqj(1),jqj(1)]), 
plot([1,4],[jqj(2),jqj(2)]), ylabel('����')
