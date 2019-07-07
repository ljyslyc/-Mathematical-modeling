%%����ģ���ۺ����۰���
clc, clear

load data.txt
sj = [repmat(data(:,1),1,3), data(:,2:end)];   % ����һ�еĳɼ�ֵҲ����ͬ���ķ�ʽ��չ
% ��ʱ����һ��Ϊһ��ָ�꣬һ��Ϊһ������

% ��һ������
n = size(sj,2)/3;
m = size(sj,1);
w = [0.5*ones(1,3), 0.125*ones(1,12)];         % ����ָ���Ȩ������(5x3)
w = repmat(w,m,1);
y = [];
for i = 1:n
    tm = sj(:,3*i-2:3*i);
    max_t = max(tm);
    max_t = repmat(max_t, m, 1);
    max_t = max_t(:,3:-1:1);
    yt = tm./max_t;
    yt(:,3) = min([yt(:,3)'; ones(1,m)]);
    y = [y,yt];
end

% ���ģ�����߾���
r = [];
for i = 1:n
    tm1 = y(:,3*i-2:3*i);
    tm2 = w(:,3*i-2:3*i);
    r = [r, tm1.*tm2];
end

% ��M+��M-�ľ���
m_plus = max(r);
m_minus = min(r);
d_plus = dist(m_plus,r');
d_minus = dist(m_minus,r');

% ��������
mu = d_minus./(d_minus+d_plus);
[mu_sort, ind] = sort(mu, 'descend')
