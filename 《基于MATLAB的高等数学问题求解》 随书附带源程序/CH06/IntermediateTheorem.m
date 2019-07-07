function xi=IntermediateTheorem(fun,range,C)
%INTERMEDIATETHEOREM   ��֤���������Ľ�ֵ����
% IntermediateTheorem(FUN,RANGE,C)  ��ͼ�ε���ʽ��֤���������ڱ������ϵĽ�ֵ����
% XI=IntermediateTheorem(FUN,RANGE,C)  ��������������һ����ֵ�㣬��������ͼ��
%
% ���������
%     ---FUN�����������ı��ʽ
%     ---RANGE��ָ��������[a,b]
%     ---C������FUN(a)��FUN(b)������ʵ��
% ���������
%     ---XI��XI����FUN(XI)=C����Ϊָ���������ͼ�η�ʽ��֤
%
% See also fzero

if nargin==2
    C=0;
end
fab=feval(fun,range);
if prod(fab-C)<=0  % �ж�C�Ƿ�����f(a)��f(b)֮��
    if fab(1)==0
        x0=range(1);
    elseif fab(2)==0
        x0=range(2);
    else
        x0=fzero(@(x)fun(x)-C,range);
    end
else
    return
end
if nargout==1  % �ж������������
    xi=x0;
else
    fplot(fun,range)
    hold on
    plot(xlim,[C,C],'k--')
    plot(x0,fun(x0),'k*')
    title(['\xi=',num2str(x0)])
end
web -broswer http://www.ilovematlab.cn/forum-221-1.html