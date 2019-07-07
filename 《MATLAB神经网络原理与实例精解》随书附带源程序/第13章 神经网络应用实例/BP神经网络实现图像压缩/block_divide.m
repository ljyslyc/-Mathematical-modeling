function P = block_divide(I,K)
% P=block_divede(I)
% [row,col]=size(I),row%K==0, and col%K==0
% divide matrix I into K*K block,and reshape to 
% a K^2*N matrix
% example:
% I=imread('lena.jpg');
% P=block_divide(I,4);

% �����ĸ�����R*C��
[row,col]=size(I);
R=row/K;
C=col/K;

% Ԥ����ռ�
P=zeros(K*K,R*C);     
for i=1:R
    for j=1:C
        % ����ȡK*K ͼ���
        I2=I((i-1)*K+1:i*K,(j-1)*K+1:j*K);
        % ��K*K���Ϊ������
        i3=reshape(I2,K*K,1);
        % ���������������
        P(:,(i-1)*R+j)=i3;
    end
end
