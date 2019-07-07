% �ű� ʹ��newff����ʵ���Ա�ʶ�� ��ȷ��
% main_newff.m

%% ����
clear,clc
rng('default')
rng(2)

%% ��������
xlsfile='student.xls';
[data,label]=getdata(xlsfile);

%% ��������
[traind,trainl,testd,testl]=divide(data,label);

%% ��������
net=feedforwardnet(3);
net.trainFcn='trainbfg';

%% ѵ������
net=train(net,traind',trainl);

%% ����
test_out=sim(net,testd');
test_out(test_out>=0.5)=1;
test_out(test_out<0.5)=0;
rate=sum(test_out==testl)/length(testl);
fprintf('  ��ȷ��\n   %f %%\n', rate*100);


