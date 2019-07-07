function df=DerivativeDefinition(fun,x,x0,type)
%DERIVATIVEDEFINITION   ���ݵ����Ķ��������ĵ���������ĳ�㴦����ֵ
% DF=DERIVATIVEDEFINITION(FUN,X)��
% DF=DERIVATIVEDEFINITION(FUN,X,[])  ����FUN����X�ĵ�����
% DF=DERIVATIVEDEFINITION(FUN,X,X0)  ����FUN�ڵ�X0���ĵ�����
% DF=DERIVATIVEDEFINITION(FUN,X,X0,TYPE)  ����TYPEָ���������������ڵ�X0���ĵ�����
%                                                 TYPE������ȡֵ��
%                                                 1.'double'��0��˫�ർ��ֵ����Ϊȱʡֵ
%                                                 2.'left'��-1������
%                                                 3.'right'��1���ҵ���
% DF=DERIVATIVEDEFINITION(FUN,X,[],TYPE)  ����TYPEָ���������������ĵ�����
%
% ���������
%     ---FUN�����ź������ʽ
%     ---X�������Ա���
%     ---X0���󵼵�
%     ---TYPE����������
% ���������
%     ---DF�����صĵ���������ֵ
%
% See also limit, diff

if nargin<4
    type=0;
end
if nargin==2 || isempty(x0)
    x0=x;
end
syms h
delta_y=subs(fun,x,x0+h)-subs(fun,x,x0);
switch type
    case {0,'double'}
        df=limit(delta_y/h,h,0);  % ����
    case {-1,'left'}
        df=limit(delta_y/h,h,0,'left');  % ������
    case {1,'right'}
        df=limit(delta_y/h,h,0,'right');  % ���ҵ���
    otherwise
        error('The Style of Derivative is Illegal.')
end
web -broswer http://www.ilovematlab.cn/forum-221-1.html