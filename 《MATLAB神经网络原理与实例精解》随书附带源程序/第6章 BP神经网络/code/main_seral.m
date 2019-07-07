% script�� main_seral.m
% ���з�ʽѵ��BP���磬ʵ���Ա�ʶ��BP���紮��ѵ��������롤

%% ����
clear all
clc

%% ��������
xlsfile='student.xls';
[data,label]=getdata(xlsfile);

%% ��������
[traind,trainl,testd,testl]=divide(data,label);

%% ���ò���
rng('default')
rng(0)
nTrainNum = 60; % 60��ѵ������
nSampDim = 2;   % ������2ά��
M=2000;         % ��������
ita=0.1;        % ѧϰ��
alpha=0.2;
%% ��������
HN=3;           % ���������
net.w1=rand(3,HN);
net.w2=rand(HN+1,1);

%% ��һ������
mm=mean(traind);
for i=1:2
    traind_s(:,i)=traind(:,i)-mm(i);
end

ml(1) = std(traind_s(:,1));
ml(2) = std(traind_s(:,2));
for i=1:2
    traind_s(:,i)=traind_s(:,i)/ml(i);
end

%% ѵ��
for x=1:M                          % ����
    ind=randi(60);                 % ��1-60��ѡһ�������
    
    in=[traind_s(ind,:),1];        % ��������
    net1_in=in*net.w1;             % ����������
    net1_out=logsig(net1_in);      % ���������
    net2_int = [net1_out,1];       % ��һ������
    net2_in = net2_int*net.w2;     % ���������
    net2_out = logsig(net2_in);    % ��������
    err=trainl(ind)-net2_out;      % ���
    errt(x)=1/2*sqrt(sum(err.^2)); % ���ƽ��
    fprintf('�� %d ��ѭ���� ��%d��ѧ���� ���  %f\n',x,ind, errt(x));
    
    % ����Ȩֵ
    for i=1:length(net1_out)+1 
        for j=1:1
            ipu1(j)=err(j);     % �ֲ��ݶ�
            % �������������֮��ĵ�����
            delta1(i,j) = ita.*ipu1(j).*net2_int(i); 
        end
        
    end
   
    for m=1:3
        for i=1:length(net1_out)
            % �ֲ��ݶ�
            ipu2(i)=net1_out(i).*(1-net1_out(i)).*sum(ipu1.*net.w2);
            % ������������֮��ĵ�����
            delta2(m,i)= ita.*in(m).*ipu2(i);
        end
    end
    
    % ����Ȩֵ
    if x==1
        net.w1 = net.w1+delta2;
        net.w2 = net.w2+delta1;
    else
        net.w1 = net.w1+delta2*(1-alpha) + alpha*old_delta2;
        net.w2 = net.w2+delta1*(1-alpha) + alpha*old_delta1;   
    end
    
    old_delta1=delta1;
    old_delta2=delta2;
end

%% ����
% �������ݹ�һ��
for i=1:2
    testd_s(:,i)=testd(:,i)-mm(i);
end

for i=1:2
    testd_s(:,i)=testd_s(:,i)/ml(i);
end

testd_s = [testd_s,ones(length(testd_s),1)];
net1_in=testd_s*net.w1;
net1_out=logsig(net1_in);
net1_out=[net1_out,ones(length(net1_out),1)];
net2_int = net1_out;
net2_in = net2_int*net.w2;
net2_out=net2_in;
% ȡ��
net2_out(net2_out<0.5)=0;
net2_out(net2_out>=0.5)=1;
rate=sum(net2_out==testl')/length(net2_out);

%% ��ʾ
fprintf('  ��ȷ��:\n    %f %%\n', rate*100);
figure(1);
plot(1:M,errt,'b-','LineWidth',1.5);
xlabel('��������')
ylabel('���')
title('BP���紮��ѵ�������')
