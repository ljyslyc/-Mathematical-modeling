% �����ȷ���
clc,clear
% һ���ο����У�m���Ƚ����У�����Ƚ�ϵ�й��ڲο����еĹ���ϵ��
load example_1.txt
% ��׼�����ݣ�ע�⼫�ԵĲ�ͬ
for i = 1:15
    example_1(i, :) = example_1(i, :) / example_1(i, 1);
end
for i = 16:17
    example_1(i, :) = example_1(i, 1) ./ example_1(i, :);
end

data = example_1
% 1����ά�ȣ��У��У�������
n = size(data, 1)
% �������ϵ��: ck = �ο��� bj = �Ƚ�
ck = data(1, :); m1 = size(ck, 1);
bj = data(2:n, :); m2 = size(bj, 1);
for i = 1:m1
    for j = 1:m2
        t(j, :) = bj(j, :) - ck(i, :);
    end
    jc1 = min(min(abs(t'))); jc2 = max(max(abs(t')));
    rho = 0.5;
    ksi = (jc1 + rho*jc2) ./ (abs(t) + rho*jc2);
    rt = sum(ksi') / size(ksi, 2);
    r(i, :) = rt;
end
r % ����ϵ������

% �����Ȱ��ս�������
[rs, rind] = sort(r, 'descend')