function y = sa_func (x)
% ���� y = 5 * sin(x1 * x2) + x1 ^2 + x2 ^2;
% ����ģ���˻��㷨����

if length(x)<2,
    y= 0;
    return;
end

x1= x(1);
x2= x(2);

y = 5*sin(x1.*x2)+x1.^2+x2.^2;
