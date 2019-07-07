% pnn_test2.m
%% ����
close all
clear,clc

%% ��������
rng(2);
a=rand(14,2)*10;					% ѵ�����ݵ�
p=ceil(a)'
tc=[3,1,1,2,1,3,2,3,2,3,3,2,2,3];		% ���

x=0:.4:11;
N=length(x);
for i=1:N
    for j=1:N
        xx(1,(i-1)*N+j) = x(i);
        xx(2,(i-1)*N+j) = x(j);
    end
end

%% ����
y = pnn_net(p,tc,xx,1);

%% ��ʾ     
plot(xx(1,y==1),xx(2,y==1),'ro');
hold on;
plot(xx(1,y==2),xx(2,y==2),'b*');
plot(xx(1,y==3),xx(2,y==3),'k+');
plot(p(1,tc==1),p(2,tc==1),'ro','LineWidth',3);
plot(p(1,tc==2),p(2,tc==2),'b*','LineWidth',3);
plot(p(1,tc==3),p(2,tc==3),'k+','LineWidth',3);
axis([0,11,0,11])
legend('��һ��','�ڶ���','������');
title('������');

