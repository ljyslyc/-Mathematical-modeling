function [ Path dist] = n2shortf( W,k1,k2)
% W��Ȩֵ����k1����ʼ�㣬k2���յ�
% Path������֮���·��
% dist������֮�����·����Ȩֵ��
% ������Floyd�㷨�����̾������
% Ȼ��ֱ����ȡ����Ҫ������֮���·��Ȩֵ֮��
% ������·��

% floyd�㷨
n = length(W);
U = W;
m = 1;
while m <= n
    for i=1:n
        for j=1:n
            if U(i,j) > U(i,m)+U(m,j)
                U(i,j)=U(i,m)+U(m,j);
            end
        end
    end
    m=m+1;
end
dist = U(k1,k2);

% ��ȡ·��
Path=zeros(1,n);
kk = k1;
Path(1)=k1;
k=1;
while kk~=k2
    for i = 1:n
        T=U(kk,k2) - W(kk,i);
        if T-U(i,k2)<10^(-8) && T-U(i,k2)>=0 && i ~= kk
            Path(k+1)=i;
            kk=i;
            k=k+1;
        end
    end
end
Path=Path(Path~=0);
end

