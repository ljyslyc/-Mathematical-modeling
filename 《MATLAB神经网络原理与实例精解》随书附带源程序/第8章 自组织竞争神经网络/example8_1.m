% example8_1.m

pos = gridtop(8,5);		% ��������
pos                     % ��Ԫ������

net = selforgmap([8 5],'topologyFcn','gridtop');
plotsomtop(net)			% ��ʾ����
web -broswer http://www.ilovematlab.cn/forum-222-1.html