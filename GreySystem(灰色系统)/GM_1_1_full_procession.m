clc, clear
x0 = [71.1 72.4 72.4 72.1 71.4 72.0 71.6];
n = length(x0);
% �󼶱�
lambda = x0(1: n-1) ./ x0(2: n)
range = minmax(lambda)   % �����������߾����е����ֵ����Сֵ���ж��Ƿ���Ҫ��Χ֮��

x1= cumsum(x0);
for i = 2:n
    z(i) = 0.5 * (x1(i) + x1(i-1));
end
B = [-z(2:n)', ones(n-1, 1)];
Y = x0(2:n)';
u  = B \ Y
x = dsolve('Dx+a*x=b', 'x(0) = x0')
x = subs(x, {'a', 'b', 'x0'}, {u(1), u(2), x1(1)});
predict_1 = double(subs(x, 't', [0:n-1]))

predict = [x0(1), diff(predict_1)]
% �в�
epsilon = x0 - predict
% ������
delta = abs(epsilon ./ x0)
% ����ƫ��
rho = 1 - (1 - 0.5 * u(1)) / (1 + 0.5 * u(1)) * lambda
