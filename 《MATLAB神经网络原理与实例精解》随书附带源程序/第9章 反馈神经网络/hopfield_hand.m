% hopfield_hand.m
%% ����
close all
clear,clc

%% 
disp('����������Ϊ:');
c1=[-1,1]						% ��һ��ƽ���
c2=[1,-1]						% �ڶ���ƽ���

% ����Ȩֵ����
w=zeros(2,2);
for i=1:2
   for j=1:2
      if (i~=j)
         w(i,j)=1/2*(c1(i)*c1(j) + c2(i)*c2(j)); 
      end
   end
end

% ��ֵ����
b=[0,0];

disp('Ȩֵ����Ϊ');
w
disp('��ֵ����Ϊ');
b

% �����ʼֵ
rng(0);
y=rand(1,2)*2-1;
y(y>0)=1;
y(y<=0)=-1;

% ѭ������
disp('��ʼ����');
while 1
    % ������һ�ε�����״ֵ̬
    disp('����״ֵ̬:');
    tmp = y
    
    % ���µ�һ����Ԫ
    y_new1 = y * w(:,1) + b(1);
    fprintf('��һ����Ԫ�� %d ����Ϊ %d \n', y(1), y_new1);
    y=[y_new1, y(2)];
    
    % ���µڶ�����Ԫ
    y_new2 = y * w(:,2) + b(2);
    fprintf('�ڶ�����Ԫ�� %d ����Ϊ %d \n', y(2), y_new2);
    y=[y(1), y_new2];
    
    % ���״ֵ̬���䣬���������
    if (tmp == y)
        break;
    end
    
end


