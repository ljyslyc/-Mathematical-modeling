% example6_1.m
x=-3:.2:3;
plot(x,x,'o')
hold on;
plot([0,0],x([8,24]),'^r','LineWidth',4)	% ��ԭʼ����Ͷ�䵽Y��
plot(zeros(1,length(x)),x,'o')
grid on  %������
title('ԭʼ����')
y=logsig(x);								% ����y��ֵ
figure(2);
plot(x,y,'o')								% ��ʾy
hold on;
plot(zeros(1,length(y)),y,'o')
plot([0,0],y([8,24]),'^r','LineWidth',4)
grid on
title('Sigmoid��������֮��')
% web -broswer http://www.huangchongqing.top