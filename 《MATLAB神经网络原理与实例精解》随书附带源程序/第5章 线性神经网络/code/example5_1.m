% example5_1.m
x=-5:5;
y=3*x-7;					% ֱ�߷���Ϊ 
randn('state',2);				% �������ӣ������ظ�ִ��
y=y+randn(1,length(y))*1.5;			% ����������ֱ��
plot(x,y,'o');
P=x;T=y;
net=newlind(P,T);				% ��newlind�������Բ�
new_x=-5:.2:5;					% �µ���������
new_y=sim(net,new_x);				% ����
hold on;plot(new_x,new_y);
legend('ԭʼ���ݵ�','��С�������ֱ��');
net.iw						% ȨֵΪ2.9219

% ans = 
% 
%     [2.9219]

net.b						% ƫ��Ϊ-6.6797

% ans = 
% 
%     [-6.6797]

title('newlind������С�������ֱ��');
web -broswer http://www.ilovematlab.cn/forum-222-1.html