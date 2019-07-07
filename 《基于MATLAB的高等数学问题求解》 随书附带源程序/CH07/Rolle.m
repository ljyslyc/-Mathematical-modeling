function varargout=Rolle(fun,range)
%ROLLE   ��֤������ĳ���������Ƿ������޶�����
% ROLLE(FUN,RANGE)  ��ͼ�εķ�ʽ��ʾ������ĳ�������ϵ��޶�����
% TF=ROLLE(FUN,RANGE)  ��ͼ�εķ�ʽ��ʾ������ĳ�������˵��޶�����
%                          �����ر��������Ƿ������޶����������TF=1���㣻TF=0������
% [TF,XI]=ROLLE(FUN,RANGE)  ���ر���������ָ���������Ƿ������޶��������TF��һ���޶���
%
% ���������
%     ---FUN��������MATLAB��������������������������������M�ļ�
%     ---RANGE��ָ��������
% ���������
%     ---TF�������Ƿ������޶��������
%     ---XI���޶���
%
% See also fzero

fab=subs(fun,range);
tf=0;
if fab(1)~=fab(2)
    disp('����fun������range�ϲ������޶�����.')
    return
else
    tf=1;
end
df=diff(fun);
while 1
    xi=fzero(inline(df),rand);
    if prod(xi-range)<=0
        break
    end
end
if nargout==2 && tf==1
    varargout{1}=tf;
    varargout{2}=xi;
else
    varargout{1}=tf;
    ezplot(fun,range)
    hold on
    plot([xi-diff(range)/10,xi+diff(range)/10],[0,0],'k--')
    title(['\xi=',num2str(xi)])
end
web -broswer http://www.ilovematlab.cn/forum-221-1.html