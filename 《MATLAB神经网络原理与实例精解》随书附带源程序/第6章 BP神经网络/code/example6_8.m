% example6_8.m
x=-4:.1:4;
y=logsig(x);			% logsig����
dy=dlogsig(x,y);        % logsig�����ĵ���
subplot(211)
plot(x,y);
title('logsig')
subplot(212);
plot(x,dy);
title('dlogsig')
web -broswer http://www.ilovematlab.cn/forum-222-1.html