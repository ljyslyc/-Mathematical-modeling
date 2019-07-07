function C=Direction_Cosine(r)
%DIRECTION_COSINE   �������ķ�������
% DIRECTION_COSINER(R)  ��������R����������λ�ù�ϵ
% C=DIRECTION_COSINE(R)  ������R�ķ�������
%
% ���������
%     ---R����������
% ���������
%     ---C�������ķ�������
%
% See also Distance, drawvec

[m,n]=size(r);
if m~=1 && n~=1
    error('�����������ʾ��ʽ����.')
end
L=Distance(r);
Cosine=r/L;
if nargout==0
    if isnumeric(Cosine) && (n==2 || n==3)
        drawvec(r)
        hold on
        drawvec([r(1),0,0])
        drawvec([0,r(2),0])
        drawvec([0,0,r(3)])
        title(['�������ң�[',num2str(Cosine),']'])
    else
        C=Cosine;
    end
else
    C=Cosine;
end
web -broswer http://www.ilovematlab.cn/forum-221-1.html