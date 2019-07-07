function xi=Lagrange(fun,range)
%LAGRANGE   ��֤������ĳ���������Ƿ���������������ֵ����
% LAGRANGE(FUN,RANGE)  ��ͼ�εķ�ʽ��ʾ������ĳ�������ϵ�����������ֵ����
% XI=LAGRANGE(FUN,RANGE)  ���غ�����ָ�������ϵ�һ������������ֵ��
%
% ���������
%     ---FUN��������MATLAB��������������������������������M�ļ�
%     ---RANGE��ָ��������
% ���������
%     ---XI������������ֵ��
%
% Sea also Rolle

fab=subs(fun,range);
df=diff(fun);
while 1
    x=fzero(inline(df-diff(fab)/diff(range)),rand);
    if prod(x-range)<=0
        break
    end
end
if nargout==1
    xi=x;
else
    ezplot(fun,range)
    hold on
    x_range=[x-diff(range)/10,x+diff(range)/10];
    y_range=diff(fab)/diff(range)*(x_range-x)+subs(fun,x);
    plot(x_range,y_range,'k--')
    title(['\xi=',num2str(x)])
end
web -broswer http://www.ilovematlab.cn/forum-221-1.html