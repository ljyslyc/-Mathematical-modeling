% example4_5.m
net=newp([0,1;-2,2],1);		% ������֪��
net.iw{1,1}			% ����ʱ��Ȩֵ

% ans =
% 
%  0     0

net.b{1}			% ����ʱ��ƫ��

% ans =
% 
%  0

P=[0,1,0,1;0,0,1,1]		% ѵ����������

% P =
% 
%  0     1     0     1
%  0     0     1     1

T=[0,0,0,1]			% ѵ�������������������

% T =
% 
%    0     0     0     1

net=train(net,P,T);		% ѵ��
net.iw{1,1}				% ѵ�����Ȩֵ
% ans =
% 
%  1     2

net.b{1}				% ѵ�����ƫ��
% ans =
% 
% -3

net=init(net);			% ��ʼ��
net.iw{1,1}				% ��ʼ�����Ȩֵ
% ans =
% 
%  0     0

net.b{1}				% ��ʼ�����ƫ��
% ans =
% 
%  0
net.initFcn				% net.initFcnֵ
% ans =
% 
% initlay
net.initParam			% ��net.initFcn= initlayʱ��net.initParam�Զ�Ϊ��
% SWITCH expression must be a scalar or string constant.
% 
% Error in network/subsref (line 140)
%         switch (subs)
% web -broswer http://www.ilovematlab.cn/forum-222-1.html