% example6_3.m
% newff
x=-4:.5:4;
y=x.^2-x;
net=newff(minmax(x),minmax(y),10);					% netΪ�°�newff������
net=train(net,x,y);									% ѵ��
xx=-4:.2:4;
yy=net(xx);
plot(x,y,'o-',xx,yy,'*-')
title('�°�newff')
net1=newff(minmax(x),[10,1],{'tansig','purelin'},'trainlm');	% net1Ϊ�ɰ�newff������
net1=train(net1,x,y);								% ѵ��
yy2=net1(xx);
figure(2);
plot(x,y,'o-',xx,yy2,'*-')
title('�ɰ�newff')
% web -broswer http://www.ilovematlab.cn/forum-222-1.html