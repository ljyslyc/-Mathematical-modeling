% example6_4.m

[x,t] = simplefit_dataset;		% MATLAB�Դ����ݣ�x��t��Ϊ1*94����
net = feedforwardnet;			% ����ǰ������
view(net)
net = train(net,x,t);			% ѵ����ȷ���������������ά��
view(net)
y = net(x);
perf = perform(net,y,t)			% �����������
web -broswer http://www.ilovematlab.cn/forum-222-1.html