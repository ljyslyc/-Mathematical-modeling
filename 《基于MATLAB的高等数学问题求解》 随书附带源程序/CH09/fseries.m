function [A,B,F,type]=fseries(f,x,n,a,b)
%FSERIES   ����Ҷ������⣬�����ظ���Ҷ����������
% [A,B,F]=FSERIES(FUN,X,N)  ����(��ż)����FUN������[-pi,pi]�ϵ�N�׸���Ҷչʽ
% [A,B,F]=FSERIES(FUN,X,N,ALPHA,BETA)  ����(��ż)����FUN��ָ�������ϵ�N�׸���Ҷչʽ
% [A,B,F,TYPE]=FSERIES(...)  �����ĸ���Ҷչʽ�����ظ���Ҷ��������
%
% ���������
%     ---FUN�������Ĵ�չ������
%     ---X���Ա���
%     ---N��չ������
%     ---ALPHA,BETA������չ�����䣬Ĭ��ֵΪ[-pi,pi]
% ���������
%     ---A,B������Ҷϵ������
%     ---F�������ĸ���Ҷչ��ʽ
%     ---TYPE������Ҷ���������ַ���
%
% See also int, fseriessym, fseriesquadl

if nargin==3
    a=-pi;b=pi;
end
L=(b-a)/2;
f1=subs(f,-x);
A=sym(zeros(1,n+1));
B=sym(zeros(1,n));
F=0;
if isequal(simple(f+f1),0)  % �溯��
    for k=1:n
        B(k)=2*int(f*sin(k*pi*x/L),x,0,L)/L;
        F=F+B(k)*sin(k*pi*x/L);
    end
    type='���Ҽ���';
elseif isequal(f,f1)  % ż����
    for k=0:n
        A(k+1)=2*int(f*cos(k*pi*x/L),x,0,L)/L;
        F=F+A(k+1)*cos(k*pi*x/L);
    end
    type='���Ҽ���';
else  % һ�㺯��
    A(1)=int(f,x,-L,L)/L;
    F=A(1)/2;
    for k=1:n
        A(k+1)=int(f*cos(k*pi*x/L),x,-L,L)/L;
        B(k)=int(f*sin(k*pi*x/L),x,-L,L)/L;
        F=F+A(k+1)*cos(k*pi*x/L)+B(k)*sin(k*pi*x/L);
    end
    type='һ�����Ǽ���';
end
web -broswer http://www.ilovematlab.cn/forum-221-1.html