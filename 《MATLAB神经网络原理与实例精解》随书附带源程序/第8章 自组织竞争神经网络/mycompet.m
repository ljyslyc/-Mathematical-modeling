% mycompet.m

%% ����
clear,clc
close all
rng(0)

%% ��������
x0=[4.1,1.8,0.5,2.9,4.0,0.6,3.8,4.3,3.2,1.0,3.0,3.6,3.8,3.7,3.7,8.6,9.1,...
    7.5,8.1,9.0,6.9,8.6,8.5,9.6,10.0,9.3,6.9,6.4,6.7,8.7;...
    8.1,5.8,8.0,5.2,7.1,7.3,8.1,6.0,7.2,8.3,7.4,7.8,7.0,6.4,8.0,...
    3.5,2.9,3.8,3.9,2.6,4.0,2.9,3.2,4.9,3.5,3.3,5.5,5.0,4.4,4.3];
[x,x_b]=mapminmax(x0);

%% ��������
net.nIN=2;
net.nCLASS =2;      % �����
net.w=rand(net.nCLASS,net.nIN);
net.it=0.2;         % ѧϰ��

%% ѵ��
N=2000;
for j=1:N
    xx=x(:,randi(30));
    net.out=net.w*xx;  
    [m,ind]=max(net.out); % ind ʤ��
    
    dw = net.it * (xx - net.w(ind,:)');
    net.w(ind,:) = net.w(ind,:) + dw';
    
end

%% ����
y=net.w*x;
[~,ind]=max(y);

%% ���
x1=x(:,ind==1);
x2=x(:,ind==2);
disp(' ���  �����')
for i=1:length(ind)
    fprintf('  %d      %d\n', i, ind(i));    
end

figure
plot(x1(1,:),x1(2,:),'ro')
hold on
plot(x2(1,:),x2(2,:),'*')

ww=net.w;
plot(ww(1,:),ww(2,:),'pk','LineWidth',2);
set(gcf,'color','w')
title('���������������')
legend('��һ��','�ڶ���','��������')
