clc,clear
syms x(t) y(t) z(t) %������ű���
X=[x;y;z]; A=[1,0,0;2,1,-2;3,2,1];B=[0;0;exp(t)*cos(2*t)];
X0=[0;1;1]; %��ֵ����
X=dsolve(diff(X)==A*X+B,X(0)==X0) %����Ž�
X=simplify([X.x;X.y;X.z]) %��ʾ��ĸ�������
pretty(X) %�����߾��е���ʾ��ʽ