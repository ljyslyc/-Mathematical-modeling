% example8_12.m

P = [-3 -2 -2  0  0  0  0 +2 +2 +3; ...		% ����������10����ά����
0 +1 -1 +2 +1 -1 -2 +1 -1  0];
Tc = [1 1 1 2 2 2 2 1 1 1];					% Ŀ�����
T = ind2vec(Tc);
net = newlvq(P,4,[.6 .4]);					% ����LVQ����
view(net)
net = train(net,P,T);						% ѵ��
Y = net(P)                                  % ����
Yc = vec2ind(Y)
Tc

web -broswer http://www.ilovematlab.cn/forum-222-1.html