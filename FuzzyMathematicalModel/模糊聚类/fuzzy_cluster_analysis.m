%% ģ����������İ���
clc, clear

% ����ģ������
load data.txt;
A = data;
[m, n] = size(A);

mu = mean(A); sigma = std(A);  % aj��bj
% ��ģ�����ƾ���
for i = 1:n
    for j = 1:n
        r(i,j) = exp(-(mu(j)-mu(i))^2 / (sigma(i)+sigma(j))^2);   % rΪģ�����ƾ���
    end
end

r1 = fuzzy_matrix_compund(r, r);
r2 = fuzzy_matrix_compund(r1, r1);
r3 = fuzzy_matrix_compund(r2, r2);   % R4�Ĵ��ݱհ���������ĵȼ۾���

b_hat = zeros(n);
lambda = 0.998;
b_hat(find(r2>lambda)) = 1;          % b_hat����ӳ�˷�����

save data1 r A
