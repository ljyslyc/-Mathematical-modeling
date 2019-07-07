function S=GraphicArea(varargin)
%GRAPHICAREA   ʹ�ö�������ƽ��ͼ�ε����
% S=GRAPHICAREA(F,G,A,B,'dicarl')  ����ֱ������ϵ������F��G��ֱ��X=A��X=B��Χͼ��
%                                         ������������ں���F��Gֻ����һ�����ű���������
% S=GRAPHICAREA(F,G,X,A,B,'dicarl')  ����ֱ������ϵ������F��G��ֱ��X=A��X=B��Χ
%                                           ͼ�ε��������ָ�������Ա���ΪX
% S=GRAPHICAREA(R,ALPHA,BETA,'polar')  ���㼫����ϵ������R��T=ALPHA��T=BETA��Χͼ��
%                                              �����������Rֻ����һ�����ű���T
% S=GRAPHICAREA(R,T,ALPHA,BETA,'polar')  ���㼫����ϵ������R��T=ALPHA��T=BETA��Χ
%                                                ͼ�ε��������ָ�������Ա���ΪT
%
% ���������
%     ---F,G��ֱ������ϵ�����ߵĺ�������
%     ---R��������ϵ�����ߵĺ�������
%     ---A,B��ֱ������ϵ�µĻ�������������
%     ---ALPHA,BETA��������ϵ�µĻ�������������
%     ---TYPE������ϵ���ͣ���'dicarl'��'polar'����ȡֵ
% ���������
%     ---S�����ص�ͼ�ε����
%
% See also int

args=varargin;
type=args{end};
switch lower(type)
    case 'dicarl'
        f1=args{1};
        f2=args{2};
        s=unique([symvar(f1),symvar(f2)]);
        if length(s)>1 || nargin==6
            x=args{3};
            a=args{4};
            b=args{5};
        else
            if nargin==5
                x=s;
                a=args{3};
                b=args{4};
            end
        end
        S=simple(int(f1-f2,x,a,b));
    case 'polar'
        r=args{1};
        s=symvar(r);
        if length(s)>1 || nargin==5
            t=args{2};
            alpha=args{3};
            beta=args{4};
        else
            if nargin==4
                t=s;
                alpha=args{2};
                beta=args{3};
            end
        end
        S=simple(1/2*int(r^2,t,alpha,beta));
    otherwise
        error('Illegal options.')
end
S=abs(S);
web -broswer http://www.ilovematlab.cn/forum-221-1.html