% digital_rec.m  ��д�����ֵ�ʶ��

%% �������ռ�
clear,clc
close all

%% ��ȡ����
disp('��ʼ��ȡͼƬ...');
I = getPicData();
% load I
disp('ͼƬ��ȡ���')

%% ������ȡ
x0 = zeros(14, 1000);
disp('��ʼ������ȡ...')
for i=1:1000
    % �Ƚ�����ֵ�˲�
    tmp = medfilt2(I(:,:,i),[3,3]);
    
    % �õ���������
    t= getFeature(tmp);
    x0(:,i) = t(:);
end

% ��ǩ label Ϊ����Ϊ1000��������
label = 1:10;
label = repmat(label,100,1);
label = label(:);
disp('������ȡ���')

%% ������ģ�͵Ľ���
tic
spread = .1;
% ��һ��
[x, se] = mapminmax(x0);
% ��������������
net = newpnn(x, ind2vec(label'));
ti = toc;
fprintf('��������ģ�͹���ʱ %f sec\n', ti);

%% ����
% ����ԭ�����������в���
lab0 = net(x);
% �������������lab0ת��Ϊ�������lab
lab = vec2ind(lab0);
% ������ȷ��
rate = sum(label == lab') / length(label);
fprintf('ѵ�������Ĳ�����ȷ��Ϊ\n  %d%%\n', round(rate*100));

%% ��������ͼƬ����
I1 = I;
% ����������ǿ��
nois = 0.2;
fea0 = zeros(14, 1000);
for i=1:1000
    tmp(:,:,i) = I1(:,:,i);
    % �������
    tmpn(:,:,i) =  imnoise(double(tmp(:,:,i)),'salt & pepper', nois);
%     tmpn(:,:,i) =  imnoise(double(tmp(:,:,i)),'gaussian',0, 0.1);
    % ��ֵ�˲�
    tmpt = medfilt2(tmpn(:,:,i),[3,3]);
    % ��ȡ��������
    t = getFeature(tmpt);
    fea0(:,i) = t(:);
end

% ��һ��
fea = mapminmax('apply',fea0, se);
% ����
tlab0 = net(fea);
tlab = vec2ind(tlab0);

% �������������µ���ȷ��
rat = sum(tlab' == label) / length(tlab);
fprintf('��������ѵ������������ȷ��Ϊ\n  %d%%\n', round(rat*100));

web -broswer http://www.ilovematlab.cn/forum-222-1.html