function tf=FunContinuity(x0,fun_left,fun_x0,fun_right)
%FUNCONTINUITY   �жϺ�����ĳ�㴦��������
% TF=FUNCONTINUITY(X0,FUN_LEFT,FUN_X0,FUN_RIGHT)  �жϷֶκ���FUN�ڵ�X0���������ԣ�
%               �������򷵻�TF=1�����򷵻�TF=0��FUN�������ұ��ʽ�Լ��ڵ�X0���ı��ʽ��ʾ
%
% ���������
%     ---X0��ָ���ĵ�
%     ---FUN_LEFT��X<X0ʱ�ĺ������ʽ
%     ---FUN_X0��X=X0ʱ�ĺ������ʽ
%     ---FUN_RIGHT��X>X0ʱ�ĺ������ʽ
% ���������
%     ---TF�������������ԣ���������X0����������TF=1������TF=0
%
% See also limit

fx0=subs(fun_x0,'x',x0);
fx0_left=limit(fun_left,'x',x0,'left');
fx0_right=limit(fun_right,'x',x0,'right');
if isequal(fx0,fx0_left) && isequal(fx0,fx0_right)
    tf=1;
else
    tf=0;
end
web -broswer http://www.ilovematlab.cn/forum-221-1.html