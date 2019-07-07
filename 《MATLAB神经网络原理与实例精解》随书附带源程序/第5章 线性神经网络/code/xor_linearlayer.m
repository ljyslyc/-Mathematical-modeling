% xor_linearlayer.m
%% ����
close all
clear,clc

%% �������
P1=[0,0,1,1;0,1,0,1]            	% ԭʼ��������
p2=P1(1,:).^2;
p3=P1(1,:).*P1(2,:);
p4=P1(2,:).^2;
P=[P1(1,:);p2;p3;p4;P1(2,:)]    	% ��ӷ����Գɷֺ����������
d=[0,1,1,0]                     	% �����������
lr=maxlinlr(P,'bias')			% �����������������ѧϰ��

%% ��������ʵ��
net=linearlayer(0,lr);          	% ������������
net=train(net,P,d);             	% ��������ѵ��


%% ��ʾ
disp('�������')                	% ���������
Y1=sim(net,P)
disp('�����ֵ���');
YY1=Y1>=0.5
disp('����Ȩֵ��')
w1=[net.iw{1,1}, net.b{1,1}]
                    
plot([0,1],[0,1],'o','LineWidth',2);    % ͼ�δ������        
hold on;
plot([0,1],[1,0],'d','LineWidth',2);
axis([-0.1,1.1,-0.1,1.1]);
xlabel('x');ylabel('y');
title('���������������������߼�');
x=-0.1:.1:1.1;y=-0.1:.1:1.1;
N=length(x);
X=repmat(x,1,N);
Y=repmat(y,N,1);Y=Y(:);Y=Y';
P=[X;X.^2;X.*Y;Y.^2;Y];
yy=net(P);
y1=reshape(yy,N,N);
[C,h]=contour(x,y,y1,0.5,'b');
clabel(C,h);
legend('0','1','���������������');



