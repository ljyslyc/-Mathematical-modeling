function s=Judgment(f,str)
%JUDGMENT   �жϺ����ĵ����Ի�͹��
% S=JUDGMENT(F,STR)
%
% ���������
%     ---F��ʵ��
%     ---STR�����������ַ���Ԫ������
% ���������
%     ---S�����ص����������ַ���
%
% See also iscellstr, cellstr

if ~iscellstr(str) || numel(str)~=2
    error('Input argument str is Illegal.')
end
if f<0
    s=str{1};
else
    s=str{2};
end
web -broswer http://www.ilovematlab.cn/forum-221-1.html