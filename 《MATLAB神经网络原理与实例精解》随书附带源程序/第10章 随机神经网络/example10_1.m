% example10_1.m   10.4.1�� simulannealbnd����

fun=@sa_func		% �������

% fun =
%
%     @sa_func
rng('default');
rng(0);
x0=rand(1,2)*4;		% ��ֵ
lb=[-4,-4];			% ��������
ub=[4,4]			% ��������

% ub =
%
%      4     4

% ����ѵ��
tic;[X,FVAL,EXITFLAG,OUTPUT] = simulannealbnd(fun,x0,lb,ub);toc

X					% ����ֵ�����Ա���ֵ

% X =
%
%    -1.0761    1.0775

FVAL				% ȫ������ֵ

% FVAL =
%
%    -2.2640

EXITFLAG			% �˳���־λ

% EXITFLAG =
%
%      1

OUTPUT			% output�ṹ��

% OUTPUT =

%      iterations: 1211
%       funccount: 1224
%         message: 'Optimization terminated: change in best function value less than options.TolFun.'
%        rngstate: [1x1 struct]
%     problemtype: 'boundconstraints'
%     temperature: [2x1 double]
%       totaltime: 0.8594
web -broswer http://www.ilovematlab.cn/forum-222-1.html