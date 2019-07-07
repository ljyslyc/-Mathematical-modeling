%% һ��ָ��ƽ����
clc, clear

yt = [50 52 47 51 49 48 51 40 48 52 51 59];
n = length(yt);
alpha = [0.2 0.5 0.8];   % ѡȡ������Ȩϵ����������Ƚ�
m = length(alpha);
yhat(1,1:m) = (yt(1)+yt(2)) / 2;  % ������ʼֵ
for i = 2:n
    yhat(i,:) = alpha*yt(i-1) + (1-alpha).*yhat(i-1,:);
end

err = sqrt(mean((repmat(yt',1,m)-yhat).^2));
y_predict = alpha*yt(n) + (1-alpha).*yhat(n,:);