%% ʹ�����ؿ��巽�������е�ö��ֵ�����ѡ��һЩ��������
% ����һ�����ڸ�ֵ���ĸ��ʲ�����

rng('shuffle')
p0 = 0;

tic % ��ʼ��ʱ

for i = 1:10^6
    x = 99 * rand(5, 1);
    x1 = floor(x);
    x2 = ceil(x);

    [f, g] = example_1(x1);
    if sum(g <= 0) == 4
        if p0 <= f
            x0 = x1;
            p0 = f;
        end
    end

    [f, g] = example_1(x2);
    if sum(g <= 0) == 4
        if p0 <= f
            x0 = x2;
            p0 = f;
        end
    end
end

x0, p0

toc % ����ʹ��ʱ��: ʱ���ѹ� 14.130765 �롣
