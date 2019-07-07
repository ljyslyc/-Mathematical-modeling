function y=logn(x,a)
%LOGN   ������׵Ķ���
% Y=LOGN(X,A)  �����ΪA������ΪX�Ķ�������������ص�Y��
%
% ���������
%     ---X������������
%     ---A�������ĵ���
% ���������
%     ---Y�����صĶ���ֵ
%
% See also log, log2, log10

if ~isequal(class(x),class(a))
    error('LOGN requires input arguments be the same class.');
end
if ~(isa([x,a],'double')||isa([x,a],'single'))
    error('LOGN requires input arguments of double or single class.');
end
switch a
    case exp(1)
        y=log(x);  % ��Ȼ����
    case 2
        y=log2(x);  % ��2Ϊ�׵Ķ���
    case 10
        y=log10(x);  % ���ö���
    otherwise
        y=log(x)/log(a);  % ���׹�ʽ�����ﻻ�׹�ʽ��bȡΪe
end
web -broswer http://www.ilovematlab.cn/forum-221-1.html