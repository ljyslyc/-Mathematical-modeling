function y = average(x)
%������Ԫ��ƽ��ֵ.
%������Ϊaverage���������Ϊһ����
%����Ϊ������ʱ����
[m,n] = size(x);
if (~((m == 1) || (n == 1)) || (m == 1 && n == 1))
    error('Input must be a vector')
end
y = sum(x)/length(x);      
end
