% example4_3.m
net=newp([-2,2;-2,2],1);	% ����һ����֪������2������ڵ㣬1������ڵ�
P=[0,0,1,1;0,1,0,1];		% ��������
T=[0,0,1,1];			% �������
net=train(net,P,T);         	% ѵ��
Y=sim(net,P)                	% ����
% Y =
% 
%      0     0     1     1
Y=net(P)                    	% ��һ�ֵõ�����������ʽ
% Y =
% 
%      0     0     1     1
% web -broswer http://www.ilovematlab.cn/forum-222-1.html