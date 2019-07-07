function varargout=PlaneEquation(varargin)
%PLANEEQUATION   ��ƽ��ķ���
% L=PLANEEQUATION(N,M0)  ƽ��ĵ㷨ʽ����
% L=PLANEEQUATION(A,B,C,D)  ƽ���һ�㷽��
% [L,TYPE]=PLANEEQUATION(...)  ��ƽ��ķ��̲����ط�������
%
% ���������
%     ---N��ƽ���ϵ�M0���ķ�����
%     ---M0��ƽ���ϵ�һ��
%     ---A,B,C,D��ƽ�淽�̵�ϵ��
% ���������
%     ---L��ƽ�淽��
%     ---TYPE��ƽ�淽�������ַ���
%
% See also dot

syms x y z
if nargin==2
    [n,M0]=deal(varargin{:});
    M=[x,y,z];
    M0M=M-M0;
    L=dot(n,M0M);
    type='ƽ��ĵ㷨ʽ����';
elseif nargin==4
    [A,B,C,D]=deal(varargin{:});
    L=A*x+B*y+C*z+D;
    type='ƽ���һ��ʽ����';
else
    error('Illegal Input arguments.')
end
L=[char(L),'=0'];
if nargout==1
    varargout{1}=L;
elseif nargout==2
    varargout{1}=L;varargout{2}=type;
else
    error('Illegal output arguments.')
end
web -broswer http://www.ilovematlab.cn/forum-221-1.html