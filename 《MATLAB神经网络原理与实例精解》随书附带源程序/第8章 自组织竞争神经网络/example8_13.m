% example8_13.m
[x,t] = iris_dataset;		% �������ݣ�xΪ����������tΪ�������
rng(0)
whos

ri=randperm(150);           % ����ѵ������Լ�
x1=x(:,ri(1:50));
t1=t(:,ri(1:50));
x2=x(:,ri(51:150));
t2=t(:,ri(51:150));
net = lvqnet(20);			% �����������ѵ��
net = train(net,x1,t1);
y = net(x2);				% ����
yy=vec2ind(y);
ty=vec2ind(t2);
sum(yy==ty)/length(yy)

web -broswer http://www.ilovematlab.cn/forum-222-1.html