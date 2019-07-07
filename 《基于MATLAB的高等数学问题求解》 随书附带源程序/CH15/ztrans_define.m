function varargout=ztrans_define(varargin)
%ZTRANS_DEFINE   ���ݶ���������z�任
% EZ=ZTRANS_DEFINE(EN,N,Z,'ztrans')  ������EN��z�任
% EN=ZTRANS_DEFINE(EZ,Z,N,'iztrans')  ��z�任ʽEZ��z��任
%
% ���������
%     ---EN,EZ�����������л�z�任ʽ�ı��ʽ
%     ---N,Z��EN��EZ�ķ����Ա���
%     ---TYPE��ָ��z�任���ͣ�������'ztrans'��'iztrans'����ȡֵ
% ���������
%     ---EZ,EN����õ�z�任ʽ��z��任ʽ
%
% See also Laplace_Define

args=varargin;
type=args{end};
switch lower(type)
    case {1,'ztrans'}
        [en,n,z]=deal(varargin{1:end-1});
        Ez=symsum(en*z^(-n),n,0,inf);
        varargout{1}=Ez;
    case {2,'iztrans'}
        [Ez,z,n]=deal(varargin{1:end-1});
        Ez=Ez*z^(n-1);
        [~,den]=numden(simple(Ez));
        zk=sort(solve(den,z));
        H=FrequencyTable(zk);
        S=H(:,1); P=H(:,2);
        R=0;
        for k=1:length(S)
            D=diff((z-zk(k))^P(k)*Ez,z,double(P(k)-1));
            R=R+1/gamma(P(k))*limit(D,z,zk(k));
        end
        varargout{1}=R;
end
web -broswer http://www.ilovematlab.cn/forum-221-1.html