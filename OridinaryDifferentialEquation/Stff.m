%% ����΢�ַ��̵����
% �߽�΢�ַ��̵Ľⷨ(�ѵ��������ת��)
clc, clear

[T, Y] = ode45('example_func2', [0, 1], [0; 1; -1])

% ���ƺ���ͼ��
plot(T, Y(:, 1), '-', T, Y(:, 2), '--')
xlabel('time t');
ylabel('solution y');
legend('y1', 'y2');