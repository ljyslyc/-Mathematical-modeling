% example6_5.m
rng('default')
rng(2)
P = [0 1 2 3 4 5 6 7 8 9 10];	% ��������
T = [0 1 2 3 4 3 2 1 2 3 4];	% �������
ff=newff(P,T,20);				% ����һ��BP���磬����һ��20���ڵ��������
ff.trainParam.epochs = 50;
ff = train(ff,P,T);				% ѵ��
Y1 = sim(ff,P);					% ����
cf=newcf(P,T,20);		        % ��newcf����ǰ������
cf.trainParam.epochs = 50;
cf = train(cf,P,T);			    % ѵ��
Y2 = sim(cf,P);					% ����
plot(P,T,'o-');					% ��ͼ
hold on;
plot(P,Y1,'^m-');
plot(P,Y2,'*-k');
title('newff & newcf')
legend('ԭʼ����','newff���','newcf���',0)
% web -broswer http://www.ilovematlab.cn/forum-222-1.html