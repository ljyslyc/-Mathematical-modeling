% grnn_test.m
%% ����
close all
clear,clc

%% ѵ������
x=-9:8;
y=[129,-32,-118,-138,-125,-97,-55,-23,-4,...
    2,1,-31,-72,-121,-142,-174,-155,-77];
P=x;
T=y;

%% ������������
xx=-9:.2:8;
yy = grnn_net(P,T,xx);

%% ��ʾ
plot(x,y,'o')
hold on;
plot(xx,yy)
hold off


