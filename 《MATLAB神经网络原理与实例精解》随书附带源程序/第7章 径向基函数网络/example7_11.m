% example7_11.m

P = [1 2 3];				% ѵ����������
T = [2.0 4.1 5.9]			% ѵ��������������ֵ
net = newgrnn(P,T);			% ���GRNN����
x=[1.5,2.5];				% �������������x=1.5��x=2.5�Ĳ���
y=sim(net,x)				% ���Խ��

web -broswer http://www.ilovematlab.cn/forum-222-1.html