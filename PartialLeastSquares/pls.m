%% ʹ��ƫ��С���˻ع��������
clc, clear

load data.txt     % ע��˴������ݵ�Ҫ��ǰ���� Ϊ�Ա�����������Ϊ�����
mu = mean(data);  % ��ֵ
sig = std(data);  % ��׼��

rr = corrcoef(data);       % ������ϵ������
std_data = zscore(data);   % ���ݱ�׼��
n = 3; m = 3;              % nΪ�Ա����ĸ�����mΪ������ĸ���
x0 = data(:,1:n);
y0 = data(:,n+1:end);
e0 = std_data(:,1:n);
f0 = std_data(:,n+1:end);
num = size(e0,1);         % ������ĸ���

chg = eye(n);              % w��w*�任����ĳ�ʼ��
for i = 1:n                
    % ����w��w*��t�ķ�����
    matrix = e0'*f0*f0'*e0;
    [vec,val] = eig(matrix);
    val = diag(val);
    [val,ind] = sort(val,'descend');
    w(:,i) = vec(:,ind(1));           % ����������ֵ��Ӧ����������
    w_star(:,i) = chg*w(:,i);       % ����w*��ȡֵ
    t(:,i) = e0*w(:,i);             % ����ɷ�ti�ĵ÷�
    alpha = e0'*t(:,i)/(t(:,i)'*t(:,i));
    chg = chg*(eye(n)-w(:,i)*alpha');
    e = e0-t(:,i)*alpha';
    e0 = e;

    % ����ss(i)��ֵ
    beta = [t(:,1:i),ones(num,1)]\f0;   % �ع鷽�̵�ϵ��
    beta(end,:) = [];                       % ɾ��ϵ���ĳ�����
    residual = f0-t(:,1:i)*beta;        % ��в����
    ss(i) = sum(sum(residual.^2));          % �����ƽ����

    % ����press(i)��ֵ
    for j = 1:num
        t1 = t(:,1:i); f1 = f0;
        discard_t = t1(j,:); discard_f = f1(j,:);    % ������ȥ�ĵ�j��������
        t1(j,:) = []; f1(j,:) = [];                  % ɾ����j���۲�ֵ
        beta1 = [t1, ones(num-1,1)]\f1;             % ��ع������ϵ��
        beta1(end,:) = [];                            % ɾ���ع�ϵ���ĳ�����
        residual = discard_f-discard_t*beta1;        % ��в�����
        press_i(j) = sum(residual.^2);
    end

    press(i) = sum(press_i);
    if i > 1
        Q_h2(i) = 1-press(i)/ss(i-1);
    else
        Q_h2(1) = 1;
    end

    if Q_h2(i) < 0.0975
        fprintf('���ɷָ���r=%d', i);
        r = i;
        break;
    end
end

beta_z = [t(:,1:r),ones(num,1)]\f0;              % ��y����t�Ļع�ϵ��
beta_z(end,:) = [];                                  % ɾ��������
coeff = w_star(:,1:r)*beta_z;                      % ��y����x�Ļع�ϵ��(�ǶԱ�׼��������ݶ��Ե�)��ÿһ��Ϊһ���ع鷽��

mu_x = mu(1:n);  mu_y = mu(n+1:end);
sig_x = sig(1:n);  sig_y = sig(n+1:end);

for i = 1:m
    ch0(i) = mu_y(i)-mu_x./sig_x*sig_y(i)*coeff(:,i); % ����ԭʼ���ݵĻع鷽��ϵ���ĳ�����
end

for i = 1:m
    coeff_origin(:,i) = coeff(:,i)./sig_x'*sig_y(i);  % ����ԭʼ���ݵĻع鷽��ϵ����ÿһ��Ϊһ���ع鷽��
end

sol = [ch0;coeff_origin];      % �ع鷽��ϵ��

% ����Ԥ��ͼ
ch0 = repmat(ch0, num, 1);
y_hat = ch0 + x0*coeff_origin;
y1_max = max(y_hat);
y2_max = max(y0);
y_max = max([y1_max;y2_max]);
residual = y_hat - y0;

subplot(2, 2, 1);
plot(0:y_max(1), 0:y_max(1), y_hat(:,1), y0(:,1), '*');

subplot(2, 2, 2);
plot(0:y_max(2), 0:y_max(2), y_hat(:,2), y0(:,2), 'O');

subplot(2, 2, 3);
plot(0:y_max(3), 0:y_max(3), y_hat(:,3), y0(:,3), 'H');