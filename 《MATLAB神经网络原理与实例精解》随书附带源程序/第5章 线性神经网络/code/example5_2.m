% example5_2.m
x=-5:5;
y=3*x-7;                        % ֱ�߷���Ϊ 
randn('state',2);		% �������ӣ������ظ�ִ��
y=y+randn(1,length(y))*1.5;	% ����������ֱ��
plot(x,y,'o');
P=x;T=y;
net=newlin(minmax(P),1,[0],maxlinlr(P));	% ��newlin������������
tic;net=train(net,P,T);toc	% ѵ������newlind��ͬ��newlin������������Ҫ����ѵ������
new_x=-5:.2:5;
new_y=sim(net,new_x);           % ����
hold on;plot(new_x,new_y);
legend('ԭʼ���ݵ�','��С�������ֱ��');
title('newlin������С�������ֱ��');
net.iw

% ans = 
% 
%     [2.9219]

net.b

% ans = 
% 
%     [-6.6797]
web -broswer http://www.ilovematlab.cn/forum-222-1.html