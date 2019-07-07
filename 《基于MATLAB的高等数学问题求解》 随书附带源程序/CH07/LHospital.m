function varargout=LHospital(num,den,x,a)
%LHOSPITAL   ��ش﷨������
% L=LHOSPITAL(NUM,DEN)��L=LHOSPITAL(NUM,DEN,[])  ��ش﷨�����NUM/DEN��0���ļ���
% L=LHOSPITAL(NUM,DEN,X)  ��ش﷨�����NUM/DEN����X=0���ļ���
% L=LHOSPITAL(NUM,DEN,X,A)  ��ش﷨�����NUM/DEN����X=A���ļ���
% [L,FORM]=LHOSPITAL(...)  ��ش﷨�����NUM/DEN�ļ��޲����ؼ���ֵL��δ��ʽ����FORM
% [L,FORM,K]=LHOSPITAL(...)  ��ش﷨�����NUM/DEN�ļ��޲����ؼ���ֵL��
%                                 δ��ʽ����FORM����ش﷨��ʹ�ô���K
%
% ���������
%     ---NUM,DEN������ʽ�ķ��Ӻͷ�ĸ���ʽ
%     ---X�������Ա���
%     ---A�����޵�
% ���������
%     ---L������ֵ
%     ---FORM��δ��ʽ���ͣ�����'��/��'��'0/0'
%     ---K����ش﷨��ʹ�ô���
%
% See also diff, subs

if nargin<4
    a=0;
end
if nargin<3 || isempty(x)
    x=unique([symvar(num),symvar(den)]);
    if length(x)>1
        error('The Symbolic variable not point out.')
    end
end
fa=subs(num,x,a);
Fa=subs(den,x,a);
if isinf(fa) && isinf(Fa)
    form='��/��';
elseif fa==0 && Fa==0
    form='0/0';
else
    error('δ��ʽ��ʽ����ȷ.')
end
k=1;
while 1
    num=diff(num);
    den=diff(den);
    fa=subs(num,x,a);
    Fa=subs(den,x,a);
    switch form
        case '��/��'
            if isinf(Fa) && ~isinf(fa)
                L=0;
                break
            end
            if ~isinf(Fa)
                L=subs(num/den,x,sym(a));
                break
            end
        case '0/0'
            if Fa==0 && fa~=0
                L=inf;
                break
            end
            if Fa~=0
                L=subs(num/den,x,sym(a));
                break
            end
    end
    k=k+1;
end
if nargout==1
    varargout{1}=L;
elseif nargout==2
    varargout{1}=L;
    varargout{2}=form;
elseif nargout==3
    varargout{1}=L;
    varargout{2}=form;
    varargout{3}=k;
else
    error('Wrong number of output arguments.')
end
web -broswer http://www.ilovematlab.cn/forum-221-1.html