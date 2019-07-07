function [tf,str]=BreakPoint(x0,fun_left,fun_x0,fun_right)
%BREAKPOINT   �жϺ�����ĳ�㴦�ļ�ϵ�����
% [TF,STR]=BREAKPOINT(X0,FUN_LEFT,FUN_X0,FUN_RIGHT)  �жϺ���FUN��X0���ļ���ԣ�
%                                                              �����ؼ�ϵ�����
%
% ���������
%     ---X0��ָ���ĵ�
%     ---FUN_LEFT��X<X0ʱ�ĺ������ʽ
%     ---FUN_X0��X=X0ʱ�ĺ������ʽ
%     ---FUN_RIGHT��X>X0ʱ�ĺ������ʽ
% ���������
%     ---TF�������������ԣ���������X0����������TF=1������TF=0
%     ---STR����ϵ������ַ�����STR����Ϊ'�����ϵ�'��'��ȥ��ϵ�'��'�񵴼�ϵ�'��
%              '��Ծ��ϵ�'��'�����ڸõ�����.'��������֮һ
%
% See also FunContinuity, limit

fx0_left=limit(fun_left,'x',x0,'left');
fx0_right=limit(fun_right,'x',x0,'right');
tf=1;
if isempty(fun_x0)
    tf=0;
else
    if isnan(fx0_left) || isnan(fx0_right) ||...  % ���޲�����
            isinf(double(fx0_left)) || isinf(double(fx0_right))
        tf=0;
    else   % ���޴���
        fx0=subs(fun_x0,'x',x0);
        if ~isequal(fx0,fx0_left) || ~isequal(fx0,fx0_right)
            tf=0;
        end
    end
end
if tf==0
    if isinf(double(fx0_left)) || isinf(double(fx0_right))  % ����Ҽ����Ƿ�Ϊ�����
        str='�����ϵ�';
    elseif isequal(fx0_left,fx0_right)  % �ж����Ҽ����Ƿ����
        str='��ȥ��ϵ�';
    elseif isnan(fx0_left) || isnan(fx0_right)  % �ж����޻��Ҽ����Ƿ����
        str='�񵴼�ϵ�';
    else
        str='��Ծ��ϵ�';
    end
else
    str='�����ڸõ�����.';
end
web -broswer http://www.ilovematlab.cn/forum-221-1.html