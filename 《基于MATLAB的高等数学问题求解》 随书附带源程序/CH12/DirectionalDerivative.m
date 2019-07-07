function dfdl=DirectionalDerivative(fun,vars,direction,M)
%DIRECTIONALDERIVATIVE   ���㷽����
% DFDL=DIRECTIONALDERIVATIVE(FUN,VARS,DIRECTION,M)  ���㺯���ڵ�M�ϵķ�����
%
% ���������
%     ---FUN����Ԫ�����ķ��ű��ʽ
%     ---VARS�������Ա���
%     ---DIRECTION����������
%     ---M��ָ���������
% ���������
%     ---DFDL�����صķ�����
%
% See also Distance, dot

if ~isa(fun,'sym')
    error('FUN must be a Symbolic function.')
end
N=length(vars);
df=sym(zeros(1,N));
for k=1:N
    df(k)=subs(diff(fun,vars{k}),vars,num2cell(M));
end
C=Direction_Cosine(direction);
dfdl=dot(df,C);
web -broswer http://www.ilovematlab.cn/forum-221-1.html