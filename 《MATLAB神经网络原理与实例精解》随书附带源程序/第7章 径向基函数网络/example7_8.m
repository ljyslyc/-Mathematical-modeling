% example7_8.m

rng(2);
a=rand(8,2)*10;			% ����ѵ��������8����ά����
p=ceil(a)

tc=[2,1,1,1,2,1,2,1];		% �������
plot(p([1,5,7],1),p([1,5,7],2),'o');
hold on;
plot(p([2,3,4,6,8],1),p([2,3,4,6,8],2),'+');
legend('��һ��','�ڶ���');
axis([0,8,1,9])
hold off
t=ind2vec(tc);
net=newpnn(p',t);		% ���PNN����
y=sim(net,p');			% ����
yc=vec2ind(y)			% ʵ����������������

web -broswer http://www.ilovematlab.cn/forum-222-1.html