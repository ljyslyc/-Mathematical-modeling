function X=dft(x,dim)
%DFT   ��ɢ����Ҷ�任
% Y=DFT(X)  �����ݾ���X����ɢ����Ҷ�任
% Y=DFT(X,DIM)  �Ծ���X����ά����ά����Ҷ�任
%
% ���������
%     ---X�����ݾ���
%     ---DIM��ָ��ά�ķ���
% ���������
%     ---Y����ɢ����Ҷ�任���
%
% See also fourier

if isvector(x)
    x=x(:).';
end
if nargin<2 || isvector(x)
    dim=1;
end
N=size(x,setdiff([1,2],dim));
n=0:N-1;
k=0:N-1;
WN=exp(-1j*2*pi/N);
nk=n'*k;
W=WN.^nk;
if dim==1
    X=x*W;
else
    X=(x.'*W).';
end
web -broswer http://www.ilovematlab.cn/forum-221-1.html