function varargout=fouriern(fun,oldvars,newvars,type,method)
%FOURIERN   ���ظ���Ҷ�任�����
% F2=FOURIERN(F1,OLDVARS,NEWVARS)  ����F1�Ķ��ظ���Ҷ�任
% F2=FOURIERN(F1,OLDVARS,NEWVARS,TYPE)  ����F1�ĸ���Ҷ�任����任��
%                                       �任������TYPEָ��
% F2=FOURIERN(F1,OLDVARS,NEWVARS,TYPE,METHOD)  ָ������fourier������任����
%                                              ����int�������
% [F2,S]=FOURIERN(...)  ����ظ���Ҷ�任�����ر任����
%
% ���������
%     ---F1����ʼ����
%     ---OLDVARS������F1�ı���
%     ---NEWVARS���任��ı���
%     ---TYPE��ָ���任���ͣ���'fourier'��'ifourier'����ȡֵ
%     ---METHOD��ָ�����任�ķ�������'fourier'��'int'���ַ���
% ���������
%     ---F2����任��ĺ���
%     ---S���任���ͣ���Ӧ��TYPE
%
% See also fourier, int

if nargin<5
    method='fourier';
end
if nargin<4 || isempty(type)
    type='fourier';
end
if ~isa(fun,'sym')
    error('FUN must be a Symbolic function.')
end
N=length(oldvars);
if length(newvars)~=N
    error('����ά����һ��.')
end
switch lower(method)
    case 'fourier'
        fcn=lower(type);
        for k=1:N
            fun=feval(fcn,fun,oldvars(k),newvars(k));
        end
    case 'int'
        if isequal(lower(type),'fourier')
            for k=1:N
                fun=int(fun*exp(-1j*oldvars(k)*newvars(k)),oldvars(k),-inf,inf);
            end
        elseif isequal(lower(type),'ifourier')
            for k=1:N
                fun=1/2/pi*int(fun*exp(1j*oldvars(k)*newvars(k)),...
                     oldvars(k),-inf,inf);
            end
        else
            error('Illegal TYPE.')
        end
    otherwise
        error('Illegal METHOD.')
end
if nargout==1
    varargout{1}=fun;
elseif nargout==2
    varargout{1}=fun;varargout{2}=[upper(type),'�任'];
end
web -broswer http://www.ilovematlab.cn/forum-221-1.html