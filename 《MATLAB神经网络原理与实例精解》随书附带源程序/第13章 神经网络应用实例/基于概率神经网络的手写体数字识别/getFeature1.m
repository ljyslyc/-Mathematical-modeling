function [Feature,bmp,flag]=getFeature1(A)
% getFeature.m
% ��ȡ64*64��ֵͼ�����������
% input:
% A: 64*64����
% output:
% Feature: ����Ϊ14����������
% bmp    : ͼ���е����ֲ���
% flag   : ��־λ����ʾ���ֲ��ֵĿ�߱�

% ��ɫ
A = ones(64) - A;

% ��ȡ���ֲ���
[x, y] = find(A == 1);

% ��ȡͼ���е����ֲ���
A = A(min(x):max(x),min(y):max(y));  

% �����߱Ⱥͱ�־λ
flag = (max(y)-min(y)+1)/(max(x)-min(x)+1);
if flag < 0.5
    flag = 0;
elseif flag >=0.5 && flag <0.75
    flag = 1;
elseif flag >=0.75 && flag <1
    flag = 2;
else
    flag = 3;
end

% ���·Ŵ󣬽���������Ϊ64
rate = 64 / max(size(A));
% �����ߴ�
A = imresize(A,rate);  
[x,y] = size(A);

% ����64�Ĳ����������
if x ~= 64
    A = [zeros(ceil((64-x)/2)-1,y);A;zeros(floor((64-x)/2)+1,y)];
end;
if y ~= 64
    A = [zeros(64,ceil((64-y)/2)-1),A,zeros(64,floor((64-y)/2)+1)];
end

%% ���������������ַ��Ľ������  F(1)~F(3)
% 1/2 ���߽�������
Vc = 32;
Num = 0;
for i = 1:64
    Num = Num+A(i, Vc);
end
F(1) = Num;
% F(1) = sum(A(:,Vc));

% 5/12 ���߽�������
Vc = round(64*3/12);
Num = 0;
for i = 1:64
    Num = Num + A(i, Vc);
end
F(2) = Num;
% F(2) = sum(A(:,Vc));

% 7/12 ���߽�������
Vc = round(64*9/12);
Num = 0;
for i = 1:64
    Num = Num + A(i, Vc);
end
F(3)=Num;
% F(3) = sum(A(:,Vc));

%% ���������������ַ��Ľ������ F(4)~F(6)
% 1/2 ˮƽ�߽�������
Hc = 32;
Num = 0;
for i = 1:64
    Num = Num + A(Hc, i);
end
F(4) = Num;
%  F(4) = sum(A(Hc,:));

% 1/3 ˮƽ�ߴ�����������
Hc = round(64/3);
Num = 0;
for i = 1:64
    Num = Num + A(Hc, i);
end
F(5) = Num;
%  F(5) = sum(A(Hc,:));
 
% 2/3ˮƽ�ߴ���������
Hc = round(2*64/3);
Num = 0;
for i = 1:64
    Num = Num + A(Hc, i);
end
F(6) = Num;
%  F(6) = sum(A(Hc,:));
%% �����Խ��ߵĽ�������
% ���Խ��߽�������
x3 = 1;
y3 = 1;
Num = 0;
for i = 0:63
    Num = Num+A(x3+i,y3+i);
end
F(7) = Num;
% F(7) = sum(diag(A));
% �ζԽ��߽�����
x4 = 1;
y4 = 64;
Num = 0;
for i = 0:63
    Num = Num + A(x4+i, y4-i);
end
F(8) = Num;
% F(8) = sum(diag(rot90(A)));
%% С����

% ���½�1/2С�����е����е�
Num = 0;
for i = 32:64
    for r = 32:64
        Num = Num + A(i,r);
    end
end
F(9) = Num/10;
% t = A(32:64,33:64);
% F(9) = sum(t(:))/10;
% ���Ͻ�1/2С�����е����е�
Num = 0;
for i3 = 1:32
    for r3 = 1:32
        Num = Num + A(i3,r3);
    end
end
F(10) = Num/10;
% t = A(1:32,1:32);
% F(10) = sum(t(:))/10;
% ���½Ƿ����е����е�
Num = 0;
for i4 = 1:32
    for r4 = 32:64
        Num = Num + A(i4,r4);
    end
end
F(11) = Num/10;
% t = A(1:32,32:64);
% F(11) = sum(t(:))/10;
% ���ϽǷ����е����е�
Num = 0;
for i5 = 32:64
    for r5 = 1:32
        Num = Num + A(i5,r5);
    end
end
F(12) = Num/10;
% t = A(32:64,1:32);
% F(12) = sum(t(:))/10;
% �·�2/3���ֵ��������ص�
Num = 0;
for i1 = 1:64
    for r1 = 16:48
        Num = Num + A(i1,r1);
    end
end
F(13) = Num/20;
% t = A(1:64,16:64);
% F(13) = sum(t(:))/20;
% �ҷ�2/3���ֵ��������ص�
Num = 0;
for i2 = 16:48
    for r2 = 1:64
        Num = Num + A(i2,r2);
    end
end
F(14) = Num/20;
% t = A(16:48,1:64);
% F(14) = sum(t(:))/20;
Feature = F';
bmp = A;
