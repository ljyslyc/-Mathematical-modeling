function L=ArcLength(varargin)
%ARCLENGTH   ����ƽ�����ߵĻ���
% L=ARCLENGTH(FUNX,FUNY,T,ALPHA,BETA,'dicarl')  ����ֱ������ϵ���ɲ�������
%                                                        ��������ƽ�����ߵĻ���
% L=ARCLENGTH(FUN,T,ALPHA,BETA,'polar')  ���㼫����ϵ����FUN�����������ߵĻ���
%
% ���������
%     ---FUNX,FUNY��ֱ������ϵ��ƽ�����ߵĲ�������
%     ---FUN��ƽ�����ߵļ����귽��
%     ---ALPHA,BETA�����ֵ�����������
%     ---TYPE������ϵ���ͣ�TYPE����������ȡֵ��
%               1.'dicarl'��'d'��1��ֱ������ϵ
%               2.'polar'��'p'��2��������ϵ
% ���������
%     ---L�����ص�ƽ�����ߵĻ���
%
% See also int

args=varargin;
type=args{end};
switch lower(type)
    case {1,'d','dicarl'}
        [funx,funy,t,alpha,beta]=deal(args{1:5});
    case {2,'p','polar'}
        [fun,t,alpha,beta]=deal(args{1:4});
        funx=fun*cos(t);
        funy=fun*sin(t);        
    otherwise
        error('Illegal options.')
end
dfx=diff(funx,t);
dfy=diff(funy,t);
L=simple(int(sqrt(dfx^2+dfy^2),t,alpha,beta));
web -broswer http://www.ilovematlab.cn/forum-221-1.html