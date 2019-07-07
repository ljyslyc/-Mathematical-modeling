%% ��ɫԤ��ģ��
clear; clc;
syms a b;
%c�д���Ҳ���
c=[a b]';
%A����֪��һЩ����
A=[];
A=input('������ҪԤ���һ������:(��[174 179...285])\n');
% ������ԣ�[174 179 183 189 207 234 220.5 256 270 285]
nnext=input('��������ҪԤ����漸�����ݣ�');
%B��ԭ����һ���ۼ���������
B=cumsum(A);
n=length(A);

%���ȼ��飬���Ƿ����ʹ�û�ɫԤ��
lamda=A(1: n-1) ./ A(2: n);
range=minmax(lamda);   % �����������߾����е����ֵ����Сֵ���ж��Ƿ���Ҫ��Χ֮��
ex='exp(-2/(n+1))';
low=subs(ex,'n',n);
ex='exp(2/(n+1))';
up=subs(ex,'n',n);
if (range(1)>=low)&&(range(2)<=up)
    disp('����ʹ�û�ɫԤ��ģ��');
end

%C���ۼӾ���(��ֵ����)
for i=1:(n-1);
    C(i)=(B(i)+B(i+1))/2;
end

%% �������������ֵ
D=A;
D(1)=[];
D=D';
E=[-C;ones(1,n-1)];
c=(E*E')\E*D;
c=c';
a=c(1)
b=c(2)

%% Ԥ���������
temp=dsolve('Dx+a*x=b', 'x(0) = x0');
temp=subs(temp, {'a', 'b', 'x0'}, {a,b,B(1)});
F=double(subs(temp, 't', 0:n+nnext-1))

%G����֪���+Ԥ����
G=[];
G(1)=A(1);
for i=2:(n+nnext)
    G(i)=F(i)-F(i-1);
end
disp('��֪���ݺ͵õ���Ԥ��������:');
disp(G);
%����ͼ��
t1=1:n;
t2=1:n+nnext;
plot(t1,A,'o',t2,G)

%% ���м���
% �в�
epsilon=A-G(1:n);
% ������
delta=abs(epsilon./A);
disp('��֪���ݺ�Ԥ������֮���������Ϊ:');
disp(delta);
% ����ƫ��
rho=1-(1-0.5*a)/(1+0.5*a)*lamda;
% ԭʼ���ݾ�ֵ
mean1=mean(A);
%ԭʼ���ݷ���
var1=var(A);
% �в�ľ�ֵ
mean2=mean(epsilon);
% �в�ķ���
var2=var(epsilon);
%�����
scale=var2/var1;
disp('�����Ϊ��(�������C0�Ƚϣ�scale<C0ʱ�ϸ�)');
disp(scale);
% С������
pp=0.6745*var1;
p=length(find(abs(epsilon)<pp))/length(epsilon);
disp('С������Ϊ��');
disp(p);
disp('�밴�ջ�ɫģ�;��ȼ�����ձ����ģ�;���');
