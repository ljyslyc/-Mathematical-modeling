function [traind,trainl,testd,testl]=divide(data,label)
% ���ݻ���
% [data,label]=getdata('student.xls')
%[traind,trainl,testd,testl]=divide(data,label)

% �����
% rng(0)
% ��Ů��ȡ30������ѵ��
TRAIN_NUM_M=30;
TRAIN_NUM_F=30;

% ��Ů�ֿ�
m_data=data(label==1,:);
f_data=data(label==0,:);

NUM_M=length(m_data); % �����ĸ���

% ��
r=randperm(NUM_M);%randperm��matlab�������������������һ���������С����ڵĲ���������������ķ�Χ
traind(1:TRAIN_NUM_M,:)=m_data(r(1:TRAIN_NUM_M),:);
testd(1:NUM_M-TRAIN_NUM_M,:)= m_data(r(TRAIN_NUM_M+1:NUM_M),:);

NUM_F=length(f_data); % Ů���ĸ���

% Ů
r=randperm(NUM_F);
traind(TRAIN_NUM_M+1:TRAIN_NUM_M+TRAIN_NUM_F,:)=f_data(r(1:TRAIN_NUM_F),:);
testd(NUM_M-TRAIN_NUM_M+1:NUM_M-TRAIN_NUM_M+NUM_F-TRAIN_NUM_F,:)=f_data(r(TRAIN_NUM_F+1:NUM_F),:);

% ��ֵ
trainl=zeros(1,TRAIN_NUM_M+TRAIN_NUM_F);
trainl(1:TRAIN_NUM_M)=1;

testl=zeros(1,NUM_M+NUM_F-TRAIN_NUM_M-TRAIN_NUM_F);
testl(1:NUM_M-TRAIN_NUM_M)=1;
