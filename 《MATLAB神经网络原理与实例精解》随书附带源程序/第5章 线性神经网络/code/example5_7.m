% example5_7.m
x=-5:5;
y=3*x-7;
randn('state',2);			% �������ӣ������ظ�ִ��
y=y+randn(1,length(y))*1.5;		% ����������ֱ��
plot(x,y,'o');
P=x;T=y;
lr=maxlinlr(P,'bias')			% �������ѧϰ��

net=linearlayer(0,lr);			% ��linearlayer�������Բ㣬�����ӳ�Ϊ0
tic;net=train(net,P,T);toc		% ��train����ѵ��
new_x=-5:.2:5;
new_y=sim(net,new_x);           	% ����
hold on;plot(new_x,new_y);
title('linearlayer������С�������ֱ��');
legend('ԭʼ���ݵ�','��С�������ֱ��');
xlabel('x');ylabel('y');
s=sprintf('y=%f * x + %f', net.iw{1,1}, net.b{1,1})

text(-2,0,s);
web -broswer http://www.ilovematlab.cn/forum-222-1.html