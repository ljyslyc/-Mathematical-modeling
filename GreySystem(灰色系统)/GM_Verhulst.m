% Grey Model Verhulst
% �����ڷǵ����İڶ���չ���У����б���״̬��S����
clc, clear
x1 = [4.93  5.33  5.87  6.35  6.63  7.15  7.37  7.39  7.81  8.35  9.39  10.59  10.94  10.44];
year = 1990:2003;
plot(year, x1, 'o-');

n = length(x1);
x0 = [x1(1), diff(x1)];
for i = 2:n
    z1(i) =  0.5 * (x1(i) + x1(i-1));
end
B = [-z1(2:end)', z1(2:end)'.^2]
Y = x0(2:end)';
abhat = B \ Y     % ��С���������������a��b��ֵ
x = dsolve('Dx+a*x=b*x^2', 'x(0)=x0');
x = vpa(subs(x, {'a', 'b', 'x0'}, {abhat(1), abhat(2), x1(1)}))
predict = double(subs(x, 't', 0:15))     % �Ա���Ĭ����t
x1_all = [x1, 9.92, 10.71];

%---���ݻ�ɫģ�;��ȼ���ȼ����м���---
% ����в�
epsilon = x1_all - predict
% ����������
delta = abs(epsilon ./ x1_all)
% ����ƽ��������
delta_mean = mean(delta)

%% ������Թ�����
% ԭʼ����ʼ���㻯
x1_all_0 = x1_all - x1_all(1);
% Ԥ������ʼ���㻯
predict_0 = predict - predict(1);
s0 = abs(sum(x1_all_0(1:end-1)) + 0.5*x1_all_0(end));
s1 = abs(sum(predict_0(1:end-1)) + 0.5*predict_0(end));
tt = predict_0 - x1_all_0;
s1_s0 = abs(sum(tt(1:end-1)) + 0.5*tt(end));
% ���Թ�����
abs_association_degree = (1 + s0 + s1) / (1 + s0 + s1 + s1_s0)

% ����������
c = std(epsilon, 1) / std(x1_all, 1)
