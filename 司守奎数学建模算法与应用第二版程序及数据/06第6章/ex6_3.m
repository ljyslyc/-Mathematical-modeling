clc,clear
syms f(x) g(x) %������ű���
df=diff(f); %����f��һ�׵��������ڳ�ֵ���ֵ�����ĸ�ֵ
[f1,g1]=dsolve(diff(f,2)+3*g==sin(x),diff(g)+df==cos(x)) %��ͨ��
f1=simplify(f1), g1=simplify(g1) %�Է��Ž���л���
[f2,g2]=dsolve(diff(f,2)+3*g==sin(x),diff(g)+df==cos(x),df(2)==0,f(3)==3,g(5)==1)
f2=simplify(f2), g2=simplify(g2) %�Է��Ž���л���
