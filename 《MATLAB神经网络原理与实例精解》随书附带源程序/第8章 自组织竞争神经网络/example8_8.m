% example8_8.m
P = [.2 .8  .1 .9; .3 .5 .4 .5];		% �����������
plot(P(1,:),P(2,:),'o');				% ���������
axis([0,1,0,1])
set(gcf,'color','w')
grid on
title('�ĸ������ķ���')
net = newc(P,2);						% ����������
 net = train(net,P);
Y = net(P)


Yc = vec2ind(Y)
P

c1=P(:,Yc==1);                          % ���Ʒ�����
c2=P(:,Yc==2);
plot(c1(1,:),c1(2,:),'ro','LineWidth',2)
hold on
plot(c2(1,:),c2(2,:),'k^','LineWidth',2)
title('�ĸ������ķ�����')
axis([0,1,0,1])
web -broswer http://www.ilovematlab.cn/forum-222-1.html