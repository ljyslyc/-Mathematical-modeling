% example9_4.m
T = [-1 -1 1; 1 -1 1]'				% ����������
% T =
% 
%     -1     1
%     -1    -1
%      1     1

net = newhop(T);					% ��newho�������
Ai = T;
[Y,Pf,Af] = net(2,[],Ai)				% ����
% Y =
% 
%     -1     1
%     -1    -1
%      1     1
% 
% 
% Pf =
%      []
% 
% 
% Af =
%     -1     1
%     -1    -1
%      1     1

net2=nnt2hop(net.LW{1},net.b{1});	% ��newho����������Ȩֵ����ֵ����net2
[Y,Pf,Af] = net(2,[],Ai)				% net2�ķ�������net��ͬ
% Y =
%     -1     1
%     -1    -1
%      1     1
% 
% 
% Pf =
%      []
% 
% 
% Af =
%     -1     1
%     -1    -1
%      1     1
web -broswer http://www.ilovematlab.cn/forum-222-1.html