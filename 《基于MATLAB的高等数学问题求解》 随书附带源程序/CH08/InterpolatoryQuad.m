function I=InterpolatoryQuad(varargin)
%INTERPOLATORYQUAD   ��ֵ�ͷ�����ⶨ����
% I=INTERPOLATORYQUAD(X,Y)  ������ɢ���ݻ���
% I=INTERPOLATORYQUAD(FUN,A,B,N)  ���㺯��FUN�ڻ�����[A,B]�ϵĻ��֣���ָ������ȷ���ΪN
%
% ���������
%     ---X,Y���۲����ݣ��ȳ�������
%     ---FUN����������
%     ---A,B���������޺�����
%     ---N������ȷ���
% ���������
%     ---I����ֵ��������
%
% See also polyfit, polyint, polyval

args=varargin;
if isnumeric(args{1})
    x=args{1};
    y=args{2};
    N=length(x)-1;
else
    [fun,a,b,N]=deal(args{:});
    h=(b-a)/N;
    x=a+h*(0:N);
    y=feval(fun,x);
end
p=polyfit(x,y,N);
P=polyint(p);
I=polyval(P,x(end))-polyval(P,x(1));
web -broswer http://www.ilovematlab.cn/forum-221-1.html