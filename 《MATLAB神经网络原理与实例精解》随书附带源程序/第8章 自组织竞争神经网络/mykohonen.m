% mykohonen.m

%% ��ջ�������
clc,clear
close all

%% ��������
x0=[4.1,1.8,0.5,2.9,4.0,0.6,3.8,4.3,3.2,1.0,3.0,3.6,3.8,3.7,3.7,8.6,9.1,...
    7.5,8.1,9.0,6.9,8.6,8.5,9.6,10.0,9.3,6.9,6.4,6.7,8.7;...
    8.1,5.8,8.0,5.2,7.1,7.3,8.1,6.0,7.2,8.3,7.4,7.8,7.0,6.4,8.0,...
    3.5,2.9,3.8,3.9,2.6,4.0,2.9,3.2,4.9,3.5,3.3,5.5,5.0,4.4,4.3];

%���ݹ�һ��
[x,m_x]=mapminmax(x0);
x=x';
[nn,mm]=size(x);

%% ����
rng(0)
%ѧϰ��
rate1max=0.8;
rate1min=0.05;
%ѧϰ�뾶
r1max=3;
r1min=0.8;

%% ���繹��
Inum=2;
% M=1;
M=2;
N=2;
K=M*N;          %Kohonen�ܽڵ���  
k=1;            %Kohonen��ڵ�����
jdpx=zeros(M*N,2);
for i=1:M
    for j=1:N
        jdpx(k,:)=[i,j];
        k=k+1;
    end
end

%Ȩֵ��ʼ��
w1=rand(Inum,K); %��һ��Ȩֵ

%% �������
ITER=200;
for i=1:ITER
    
    %����Ӧѧϰ�ʺ���Ӧ�뾶
    rate1=rate1max-i/ITER*(rate1max-rate1min);
    r=r1max-i/ITER*(r1max-r1min);
    
    %�����ȡһ������
    k=randi(30);
    xx=x(k,:);
    
    %�������Žڵ�
    [mindist,index]=min(dist(xx,w1));
    
    %��������
    d1=ceil(index/4);
    d2=mod(index,4);
    nodeindex=find(dist([d1,d2],jdpx')<r);
    
    %���ǹ���
    for j=1:K
        if sum(nodeindex==j)
            w1(:,j)=w1(:,j)+rate1*(xx'-w1(:,j));
        end
    end
end

%% ����
Index=zeros(1,30);
for i=1:30
    [mindist,Index(i)]=min(dist(x(i,:),w1));
end

%% ��ʾ
x1=x0(:,Index==1);
x2=x0(:,Index==2);
x3=x0(:,Index==3);
x4=x0(:,Index==4);
plot(x1(1,:),x1(2,:),'ro');hold on
plot(x2(1,:),x2(2,:),'k*');
plot(x3(1,:),x3(2,:),'b>');
plot(x4(1,:),x4(2,:),'mp');
title('������')
legend('���1','���2','���3','���4')
% legend('���1','���2')
set(gcf,'color','w')
box on

