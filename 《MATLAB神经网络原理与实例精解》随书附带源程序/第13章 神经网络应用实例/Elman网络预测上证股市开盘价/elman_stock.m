% elman_stock.m
%% ��������ռ��еı�����ͼ��
clear,clc
close all

% ����337����ָ֤�����̼۸�
load elm_stock

whos

%% 2.����������
% ���ݸ���
n=length(price);

% ȷ��priceΪ������
price=price(:);

% x(n) ��x(n-1),x(n-2),...,x(n-L)��L����Ԥ��õ�.
L = 6;

% price_n��ÿ��Ϊһ��������ϵ���������n-L������
price_n = zeros(L+1, n-L);
for i=1:n-L
    price_n(:,i) = price(i:i+L);
end


%% ����ѵ������������
% ��ǰ280�����ݻ���Ϊѵ������
% ��51�����ݻ���Ϊ��������

trainx = price_n(1:6, 1:280);
trainy = price_n(7, 1:280);

testx = price_n(1:6, 290:end);
testy = price_n(7, 290:end);


%% ����Elman������

% ����15����Ԫ��ѵ������Ϊtraingdx
net=elmannet(1:2,15,'traingdx');

% ������ʾ����
net.trainParam.show=1;

% ����������Ϊ2000��
net.trainParam.epochs=2000;

% ������ޣ��ﵽ�����Ϳ���ֹͣѵ��
net.trainParam.goal=0.00001;

% �����֤ʧ�ܴ���
net.trainParam.max_fail=5;

% ��������г�ʼ��
net=init(net);

%% ����ѵ��

%ѵ�����ݹ�һ��
[trainx1, st1] = mapminmax(trainx);
[trainy1, st2] = mapminmax(trainy);

% ������������ѵ��������ͬ�Ĺ�һ������
testx1 = mapminmax('apply',testx,st1);
testy1 = mapminmax('apply',testy,st2);

% ����ѵ����������ѵ��
[net,per] = train(net,trainx1,trainy1);

%% ���ԡ������һ��������ݣ��ٶ�ʵ��������з���һ��

% ��ѵ����������������в���
train_ty1 = sim(net, trainx1);
train_ty = mapminmax('reverse', train_ty1, st2);

% ��������������������в���
test_ty1 = sim(net, testx1);
test_ty = mapminmax('reverse', test_ty1, st2);

%% ��ʾ���
% ��ʾѵ�����ݵĲ��Խ��
figure(1)
x=1:length(train_ty);

% ��ʾ��ʵֵ
plot(x,trainy,'b-');
hold on
% ��ʾ����������ֵ
plot(x,train_ty,'r--')

legend('�ɼ���ʵֵ','Elman�������ֵ')
title('ѵ�����ݵĲ��Խ��');

% ��ʾ�в�
figure(2)
plot(x, train_ty - trainy)
title('ѵ�����ݲ��Խ���Ĳв�')

% ��ʾ�������
mse1 = mse(train_ty - trainy);
fprintf('    mse = \n     %f\n', mse1)

% ��ʾ������
disp('    �����')
fprintf('%f  ', (train_ty - trainy)./trainy );
fprintf('\n')

figure(3)
x=1:length(test_ty);

% ��ʾ��ʵֵ
plot(x,testy,'b-');
hold on
% ��ʾ����������ֵ
plot(x,test_ty,'r--')

legend('�ɼ���ʵֵ','Elman�������ֵ')
title('�������ݵĲ��Խ��');

% ��ʾ�в�
figure(4)
plot(x, test_ty - testy)
title('�������ݲ��Խ���Ĳв�')

% ��ʾ�������
mse2 = mse(test_ty - testy);
fprintf('    mse = \n     %f\n', mse2)

% ��ʾ������
disp('    �����')
fprintf('%f  ', (test_ty - testy)./testy );
fprintf('\n')

