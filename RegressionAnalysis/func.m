function [ yhat ] = func( beta, x )
%FUNC �����ڷ����Իع��ģ�ͺ���
%   �˴���ʾ��ϸ˵��
yhat = (beta(4) * x(:, 2) - x(:, 3) / beta(5)) ./ (1 + beta(1) * x(:, 1)+ ...
    beta(2) * x(:, 2) + beta(3) * x(:, 3));

end

