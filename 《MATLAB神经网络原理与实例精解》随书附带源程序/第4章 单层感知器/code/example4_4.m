% example4_4.m
figure;
subplot(2,1,1);
n = -5:0.01:5;
plot(n,hardlim(n),'LineWidth',2);		% hardlim����ֵ
subplot(2,1,2);
plot(n,hardlims(n),'r','LineWidth',2)		% hardlims����ֵ
title('hardlims');
subplot(2,1,1);
title('hardlim');
% web -broswer http://www.ilovematlab.cn/forum-222-1.html