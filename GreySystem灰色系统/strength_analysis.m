% ���Ʒ���
clc,clear
% m���ο����У�n���Ƚ����У�����Ƚ�ϵ�й��ڲο����еĹ���ϵ��
% �õ���������
load example_2.txt
data = example_2
n = size(data, 1);

% ��׼�����ݣ����Ծ���ͬ
for i = 1:n
    data(i, :) = data(i, :) / data(i, 1);
end

ck = data(6:n, :); m1 = size(ck, 1);
bj = data(1:5, :); m2 = size(bj, 1);
for i = 1:m1
    for j = 1:m2
        t(j, :) = bj(j, :) - ck(i, :)
    end
    jc1 = min(min(abs(t'))); jc2 = max(max(abs(t')));
    rho = 0.5;
        ksi = (jc1 + rho*jc2) ./ (abs(t) + rho*jc2);
    rt = sum(ksi') / size(ksi, 2);
    r(i, :) = rt;
end

% ���նԹ�������ķ���
r