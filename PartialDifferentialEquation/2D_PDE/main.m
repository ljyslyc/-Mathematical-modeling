%% ��ά״̬�ռ��ϵ�ƫ΢�ַ������(���ʹ��pdetool����������������)
% �����������ϵ��ȴ�������
clc, clear

% ���ⶨ��
g = 'squareg';   % ��������������
b = 'squareb1';  % �߽���Ϊ0����
c = 1; a = 0; f = 0; d = 1;

% ������ʼ������������
[p, e, t] = initmesh(g);

% �����ʼ����
u0 = zeros(size(p, 2), 1);
ix = find(sqrt(p(1,:).^2 + p(2,:).^2) < 0.4);
u0(ix) = 1;

% ��ʱ���0~0.1�����
nframe = 20;
tlist = linspace(0, 0.1, nframe);
u1 = parabolic(u0, tlist, b, p, e, t, c, a, f, d);

% ������ʾ������
for j = 1:nframe
    pdesurf(p, t, u1(:,j));
    mv(j) = getframe;
end
movie(mv, 10)