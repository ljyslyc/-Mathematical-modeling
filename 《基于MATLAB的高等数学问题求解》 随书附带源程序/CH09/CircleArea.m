function A=CircleArea(R,n)
%CIRCLEAREA   Բ��������ޱƽ�����
% A=CIRCLEAREA(R,N)  ���ö����������ޱƽ�Բ�����
%
% ���������
%     ---R��Բ�İ뾶
%     ---N��������α���
% ���������
%     ---A��Բ�Ľ������
%
% See also symsum

M=R;
A=sqrt(3)/4*M^2*6;
for k=2:n
    G=sqrt(R^2-(M/2)^2);
    j=R-G;
    m=sqrt((M/2)^2+j^2);
    a=1/2*M*j*3*2^(k-1);
    M=m;
    A=A+a;
end
if isa(R,'sym')
    A=simple(A);
end
web -broswer http://www.ilovematlab.cn/forum-221-1.html