function H=FrequencyTable(X)
%FREQUENCYTABLE   ͳ������Ԫ�س��ֵ�Ƶ��
% H=FREQUENCYTABLE(X)  ͳ�ƾ���X�и�Ԫ�س��ֵ�Ƶ��
%
% ���������
%     ---X����������������
% ���������
%     ---H�����ص�ͳ�ƽ��
%
% See also tabulate

if ~isa(X,'sym')
    H=tabulate(X);
    H=H(:,1:2);
else
    sortX=sort(X(:));
    D=[simple(sortX(2:end)-sortX(1:end-1));sym(1)];
    uniqueX=(D~=0);
    k=find([1;uniqueX]);
    H=[sortX(uniqueX) diff(k)];
end
web -broswer http://www.ilovematlab.cn/forum-221-1.html