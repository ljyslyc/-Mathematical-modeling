% example7_2.m
tic
P=-2:.2:2;
rand('state',pi);
T=P.^2+rand(1,length(P));	% �ڶ��κ����м�������
net=newrbe(P,T,3);			% �����ϸ�ľ������������
test=-2:.1:2;
out=sim(net,test);			% �������
toc
figure(1);
plot(P,T,'o');
hold on;
plot(test,out,'b-');
legend('���������','��ϵĺ���');
