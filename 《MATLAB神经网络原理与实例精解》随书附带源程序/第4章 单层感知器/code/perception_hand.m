% perception_hand.m  demo ʵ��
%% ����
clear,clc
close all

%%
n=0.2;                  % ѧϰ��
w=[0,0,0]; 
P=[ -9,  1, -12, -4,   0, 5;...
   15,  -8,   4,  5,  11, 9];
d=[0,1,0,0,0,1];        % �������

P=[ones(1,6);P];
MAX=20;                 % ����������Ϊ20��
%% ѵ��
i=0;
while 1
    v=w*P; 
    y=hardlim(v);       % ʵ�����
    %����
    e=(d-y);
    ee(i+1)=mae(e);
    if (ee(i+1)<0.001)   % �ж�
        disp('we have got it:');
        disp(w);
        break;
    end
    % ����Ȩֵ��ƫ��
    w=w+n*(d-y)*P';
    
    if (i>=MAX)         % �ﵽ�������������˳�
        disp('MAX times loop');
        disp(w);
        disp(ee(i+1));
       break; 
    end
    i= i+1;
end


%% ��ʾ
figure;
subplot(2,1,1);         % ��ʾ������ĵ�ͷ�����
plot([-9 ,  -12  -4    0],[15, 4   5   11],'o');
hold on;
plot([1,5],[-8,9],'*');
axis([-13,6,-10,16]);
legend('��һ��','�ڶ���');
title('6�������Ķ�����');
x=-13:.2:6;
y=x*(-w(2)/w(3))-w(1)/w(3);
plot(x,y);
hold off;

subplot(2,1,2);         % ��ʾmaeֵ�ı仯
x=0:i;
plot(x,ee,'o-');
s=sprintf('mae��ֵ(��������:%d)', i+1);
title(s);
