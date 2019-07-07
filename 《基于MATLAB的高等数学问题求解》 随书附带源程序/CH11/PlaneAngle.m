function theta=PlaneAngle(PI1,PI2)
%PLANEANGLE   ����ƽ��ļн�
% T=PLANEANGLE(PI1,PI2)  ��ƽ��PI1��PI2�ļн�
%
% ���������
%     ---PI1,PI2����ƽ���ϵ������
% ���������
%     ---T�����ص�ƽ��ļн�
%
% See also subspace

if isa([PI1;PI2],'sym')
    PI1=[diff(PI1,'x'),diff(PI1,'y'),diff(PI1,'z')];
    PI2=[diff(PI2,'x'),diff(PI2,'y'),diff(PI2,'z')];
end
if isvector(PI1) && isvector(PI2)
    if length(PI1)==3 && length(PI2)==3
        theta=subspace(PI1(:),PI2(:));
    else
        error('������������Ϊ��ά����.')
    end
else
    error('Illegal Input arguments.')
end
web -broswer http://www.ilovematlab.cn/forum-221-1.html