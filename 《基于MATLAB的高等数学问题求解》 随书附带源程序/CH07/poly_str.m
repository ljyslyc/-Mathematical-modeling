function d=poly_str(xd,yd,xi,N)
%POLY_STR   ��ֵ�����㷨
% D=POLY_STR(XD,YD,XI,N)  ��������XD,YD���ж���ʽ��ֵ�������ڵ�XI����N�׵���
%
% ���������
%     ---XD,YD��ʵ������
%     ---XI����ֵ�󵼵�
%     ---N���󵼽״�
% ���������
%     ---D��N����ֵ����
%
% See also diff, polyfit, polyder, polyval

L=length(xd)-1;
p=polyfit(xd,yd,L);
for k=1:N
    p=polyder(p);
end
d=polyval(p,xi);
web -broswer http://www.ilovematlab.cn/forum-221-1.html