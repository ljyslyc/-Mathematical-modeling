% example7_3.m
n = -5:0.1:5;
a = radbas(n-2);		% ����λ������ƽ��������λ
b = exp(-(n).^2/2);     % ����2�����߸��ӡ����֡�
figure;
plot(n,a);
hold on;
plot(n,b,'--');			% ����
c = diff(a);			% ����a��΢��
hold off;
figure;
plot(c);
