%% simulated_annealing
% ģ���˻��㷨��������Ѱ��ȫ����Сֵ���Ż�����

clc, clear

load example_1.txt
data = example_1;

x = data(:, 1:2:8); x = x(:);
y = data(:, 2:2:8); y = y(:);
[n, m] = size(data);         % �������ݵ�����������(��Ҫ����Ҫ����)
n = n * 4;                           % ÿһ��4�����ݣ�һ����4n���㣨�ټ���ʼĩ�� һ��4n+2���㣩

data = [x y];
d1 = [70, 40];
data = [d1; data; d1];     % ʼ�����յ�
data = data * pi / 180;   % �ӽǶ���ת��Ϊ������

distance = zeros(n+2);  % �����������
for i = 1:n+1
    for j = i + 1:n+2
        tmp = cos(data(i, 1) - data(j, 1)) * cos(data(i, 2)) * cos(data(j, 2)) + ...
        sin(data(i, 2)) * sin(data(j, 2));
        distance(i, j) = 6370 * acos(tmp);
    end
end

distance = distance + distance';
S0 = [ ]; 
sum = inf;   % ��ʵ����������޸�Ϊһ���ϴ������

% ʹ�����ؿ���ģ�����һ����ʼ��
rand('twister',5489);
for j = 1:1000
    S = [1, 1+randperm(n), n+2];            % ����Ϊn+2����������һλ��1���м���2��n+1��������У����һλ��n+2
    tmp = 0;
    for i = 1:n+1
        tmp = tmp + distance(S(i), S(i+1));
    end
    if tmp < sum
        S0 = S; sum = tmp;
    end
end

% �趨��ʼֵ
e = 0.1^30;       % ��ֹ�¶�
L = 20000;         % ѭ������
alpha = 0.999;  % ����ϵ��
T = 1;                  % ��ʼ�¶�

% �˻����
for k = 1:L
    % �����½�
    c = 2 + floor(n * rand(1, 2));   % �������һ��(2, n+2)֮����������
    c = sort(c);
    c1 = c(1); c2 = c(2);
    % ���㺯������
    delta = distance(S0(c1-1), S0(c2)) + distance(S0(c1), S0(c2+1)) - ...
    distance(S0(c1-1), S0(c1)) - distance(S0(c2), S0(c2+1));
    % ����׼��
    if delta < 0  || exp(-delta / T) > rand(1)
        S0 = [S0(1:c1-1), S0(c2:-1:c1), S0(c2+1:n+2)];
        sum = sum + delta;
    end

    T = T * alpha;
    if T < e
        break;
    end
end

% ���Ѳ��·���Լ�·������
S0, sum
data = data * 180 / pi;    % �ӻ�����ת���ؽǶ���
hold on
for i = 1: n + 1
    scatter(data(S0(i), 1), data(S0(i), 2), 'k');
    plot([data(S0(i), 1), data(S0(i+1), 1)],[data(S0(i), 2), data(S0(i+1), 2)]);
end
plot([data(S0(n+1), 1), data(S0(n+2), 1)],[data(S0(n+1), 2), data(S0(n+2), 2)]);
