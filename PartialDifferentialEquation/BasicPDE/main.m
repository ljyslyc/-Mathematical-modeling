%% ������һάƫ΢�ַ������
clc, clear

m = 0;
x = linspace(0, 1, 20); % xȡ20����
t = linspace(0, 2, 20); % ʱ��tȡ20�������

sol = pdepe(m, @pdefun, @ic, @bc, x, t);
u = sol(:, :, 1);        % ȡ����

% ��ͼ���
figure(1)
surf(x, t, u)
title('pde��ֵ��')
xlabel('λ��x')
ylabel('ʱ��t')
zlabel('��ֵ��u')

% �������Ƚ�
figure(2)
surf(x, t, exp(-t)'*sin(pi*x));
title('������')
xlabel('λ��x')
ylabel('ʱ��t')
zlabel('��ֵ��u')

% ��ʾ�ض����ϵĽ�(ָ��x��t)
figure(3)
M = length(t);   % ��ʾʱ���յ��ϵĽ�
xout = linspace(0, 1, 100);   % ������λ��
[uout, dudx] = pdeval(m, x, u(M,:), xout);
plot(xout, uout);
title('ĩʱ��ʱ��λ���µĽ�');
xlabel('x')
ylabel('u')
