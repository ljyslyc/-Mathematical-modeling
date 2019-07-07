function [x,iter,exitflag]=Equ_iter(varargin)
%EQU_ITER   �����������Է�����Ľ�
% X=EQU_ITER(A,B,'jacobi')  �ſ˱ȵ����������Է�����AX=B�Ľ�X����ʼ������X0��
%                           ����EPS������������ITER_MAX��ȡĬ��ֵ
% X=EQU_ITER(A,B,X0,EPS,ITER_MAX,'jacobi')  �ſ˱ȵ����������Է�����AX=B�Ľ�X
% [X,ITER]=EQU_ITER(...)  �ſ˱ȵ����������Է�����ĽⲢ���ص�������
% [X,ITER,EXITFLAG]=EQU_ITER(...)  �ſ˱ȵ����������Է�����ĽⲢ���ص��������ͳɹ���־
% X=EQU_ITER(A,B,'seidel')  ��˹���¶����������Է�����AX=B�Ľ�X����ʼ������X0��
%                           ����EPS������������ITER_MAX��ȡĬ��ֵ
% X=EQU_ITER(A,B,X0,EPS,ITER_MAX,'seidel')  ��˹���¶����������Է�����AX=B�Ľ�X
% [X,ITER]=EQU_ITER(...,'seidel')  ��˹���¶����������Է�����AX=B�Ľ�X�����ص�������
% [X,ITER,EXITFLAG]=EQU_ITER(...,'seidel')  ��˹���¶����������Է�����AX=B�Ľ�X��
%                                           �����ص��������ͳɹ���־
% X=EQU_ITER(A,B,W,'sor')  SOR���������Է�����AX=B�Ľ�X����ʼ������X0��
%                          ����EPS������������ITER_MAX��ȡĬ��ֵ
% X=EQU_ITER(A,B,W,X0,EPS,ITER_MAX,'sor')  SOR���������Է�����AX=B�Ľ�X
% [X,ITER]=EQU_ITER(...,'sor')  SOR���������Է�����AX=B�Ľ�X�����ص�������
% [X,ITER,EXITFLAG]=EQU_ITER(...,'sor')  SOR���������Է�����AX=B�Ľ�X��
%                                        �����ص��������ͳɹ���־
%
% ���������
%     ---A�����Է������ϵ������
%     ---B�����Է�������Ҷ���
%     ---W�����ɳ�����
%     ---X0����ʼ������Ĭ��ֵΪ������
%     ---EPS������Ҫ��Ĭ��ֵΪ1e-6
%     ---ITER_MAX��������������Ĭ��ֵΪ100
%     ---TYPE�������������ͣ�TYPE�����¼���ȡֵ��
%              1.'jacobi'��1���ſ˱ȵ�����
%              2.'seidel'��2����˹���¶�������
%              3.'sor'��3��SOR������
% ���������
%     ---X�����Է�����Ľ��ƽ�
%     ---ITER����������
%     ---EXITFLAG�������ɹ����ı�־��1��ʾ�����ɹ���0��ʾ����ʧ��
% 
% See also Gauss

args=varargin;
style=args{end};
A=args{1};
b=args{2};
[m,n]=size(A);
if m~=n || length(b)~=m
    error('���Է������ϵ������ͳ�����ά����ƥ��.')
end
iter=0;
exitflag=1;
D=diag(diag(A));
L=tril(A,-1);
U=triu(A,1);
switch lower(style)
    case {1,'jacobi'}  % Jacobi������
        if nargin==3
            x0=zeros(n,1);
            eps=1e-6;
            iter_max=100;
        elseif nargin==6
            [x0,eps,iter_max]=deal(args{3:5});
        else
            error('���������������.')
        end
        J=-inv(D)*(L+U);f=D\b;
        while iter<iter_max
            x=J*x0+f;
            if norm(x-x0,inf)<eps
                break
            end
            x0=x;iter=iter+1;
        end
    case {2,'seidel'}  % Gauss-Seidel������
        if nargin==3
            x0=zeros(n,1);
            eps=1e-6;
            iter_max=100;
        elseif nargin==6
            [x0,eps,iter_max]=deal(args{3:5});
        else
            error('���������������.')
        end
        G=-inv(D+L)*U;f_G=(D+L)\b;
        while iter<iter_max
            x=G*x0+f_G;
            if norm(x-x0,inf)<eps
                break
            end
            x0=x;iter=iter+1;
        end
    case {3,'sor'}  % SOR������
        w=args{3};
        if nargin==4
            x0=zeros(n,1);
            eps=1e-6;
            iter_max=100;
        elseif nargin==7
            [x0,eps,iter_max]=deal(args{4:6});
        else
            error('���������������.')
        end
        S=(D+w*L)\((1-w)*D-w*U);f_w=w*((D+w*L)\b);
        while iter<iter_max
            x=S*x0+f_w;
            if norm(x-x0,inf)<eps
                break
            end
            x0=x;iter=iter+1;
        end
    otherwise
        error('Illegal options.')
end
if iter==iter_max
    exitflag=0;
end
web -broswer http://www.ilovematlab.cn/forum-221-1.html