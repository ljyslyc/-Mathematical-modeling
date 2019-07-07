% example5_4.m
%% ����
clear,clc
close all

%% ��������
P=-5:5;                         % ���룺11������
d=3*P-7;
randn('state',2);	
d=d+randn(1,length(d))*1.5     % ����������������������Ժ���

P=[ones(1,length(P));P]        % P����ƫ��
lp.lr = 0.01;                  % ѧϰ��
MAX = 150;                     % ����������
ep1 = 0.1;                     % ��������ֹ��ֵ
ep2 = 0.0001;                  % Ȩֵ�仯��ֹ��ֵ
%% ��ʼ��
w=[0,0];
 
%% ѭ������
 for i=1:MAX
    fprintf('��%d�ε�����\n', i)
    e=d-purelin(w*P);          % ����������
    ms(i)=mse(e);              % ������
    ms(i)
    if (ms(i) < ep1)           % ���������С��ĳ��ֵ�����㷨����
        fprintf('������С��ָ��������ֹ\n');
       break; 
    end
    
    dW = learnwh([],P,[],[],[],[],e,[],[],[],lp,[]);    % Ȩֵ������
    if (norm(dW) < ep2)        % ���Ȩֵ�仯С��ָ��ֵ�����㷨����
       fprintf('Ȩֵ�仯С��ָ��������ֹ\n');
       break;
    end
    w=w+dW                     % ��dW����Ȩֵ    
    
 end
 
%% ��ʾ
fprintf('�㷨�����ڣ�\nw= (%f,%f),MSE: %f\n', w(1), w(2), ms(i));
figure;
subplot(2,1,1);                % ����ɢ���ֱ��
plot(P(2,:),d,'o');title('ɢ����ֱ����Ͻ��');
xlabel('x');ylabel('y');
axis([-6,6,min(d)-1,max(d)+1]);
x1=-5:.2:5;
y1=w(1)+w(2)*x1;
hold on;plot(x1,y1);
subplot(2,1,2);                % ���ƾ������½�����
semilogy(1:i,ms,'-o');
xlabel('��������');ylabel('MSE');title('�������½�����');
web -broswer http://www.ilovematlab.cn/forum-222-1.html
