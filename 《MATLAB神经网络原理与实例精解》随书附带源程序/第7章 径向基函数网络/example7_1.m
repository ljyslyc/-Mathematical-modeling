P=1:.5:10;
rand('state',pi);
T=sin(2*P)+rand(1,length(P));				  % �����Һ���������
plot(P,T,'o')
% net=newrb(P,T);
net=newrb(P,T,0,0.6);
test=1:.2:10;
out=sim(net,test);                            % ���µ�����ֵtest�������Ӧ�ĺ���ֵ
figure(1);hold on;plot(test,out,'b-');
legend('���������','��ϵĺ���');
