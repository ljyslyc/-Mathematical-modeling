function I=CoordinateCurveInt(fun,vars,fun_para,t,alpha,beta)
%COORDINATECURVEINT   ����ڶ������߻���
% I=COORDINATECURVEINT(FUN,VARS,FUN_PARA,T,ALPHA,BETA)  ���㺯��FUN�ĵڶ������߻���
%
% ���������
%     ---FUN��������������
%     ---VARS�����ű���
%     ---FUN_PARA���������ߵĲ������̵ķ��ű��ʽ
%     ---T���������̵ķ����Ա���
%     ---ALPHA,BETA����������
% ���������
%     ---I���ڶ������߻���ֵ
%
% See also diff, int

N=length(fun);
S=0;
for k=1:N
    df=diff(fun_para(k),t);
    S=S+subs(fun(k),vars,num2cell(fun_para))*df;
end
I=int(S,t);
I=subs(I,t,sym(beta))-subs(I,t,sym(alpha));
web -broswer http://www.ilovematlab.cn/forum-221-1.html