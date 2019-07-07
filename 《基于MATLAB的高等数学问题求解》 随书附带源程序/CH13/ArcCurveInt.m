function I=ArcCurveInt(fun,vars,varargin)
%ARCCURVEINT   �����һ�����߻���
% I=ARCCURVEINT(FUN,{'X','Y'},FUNX,FUNY,T,ALPHA,BETA)  �����Ԫ�����ĵ�һ�����߻���
% I=ARCCURVEINT(FUN,{'X','Y','Z'},FUNX,FUNY,FUNZ,T,ALPHA,BETA)  ������Ԫ������
%                                                               ��һ�����߻���
% I=ARCCURVEINT(FUN,{'X','Y','Z',...},FUNX,FUNY,FUNZ,...,T,ALPHA,BETA)
%                                                �����Ԫ�����ĵ�һ�����߻���
%
% ���������
%     ---FUN����������
%     ---VARS�����������ķ��ű���
%     ---FUNX,FUNY,...���������ߵĲ�������
%     ---T���������̵ķ����Ա���
%     ---ALPHA,BETA�����ַ�Χ
% ���������
%     ---I�����߻���ֵ
%
% See also diff, int

args=varargin;
[t,alpha,beta]=deal(args{end-2:end});
S=0;
for k=1:nargin-5
    fun=subs(fun,vars{k},args{k});
    df=diff(args{k},t);
    S=S+df^2;
end
I=int(fun*sqrt(S),t,alpha,beta);
web -broswer http://www.ilovematlab.cn/forum-221-1.html