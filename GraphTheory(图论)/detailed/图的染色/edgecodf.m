function [k C] = edgecodf(M)
% ��Ⱦɫ�㷨
% M��ʾ����ͼ���ڽӾ���
% k��ʾȾɫ��
% C��ʾȾɫ����

 fprintf('����Ļ���˼��Ͳ���˵�������help edgecodf');
W = M;
n = size(M,1);
% W2 = W;
for i = 1:n
    for j = 1:i-1
        W(i,j) = 0;
    end
end
es = sum(sum(W));
C = zeros(1,es);
k = 1;
Wm = W;
for i = 1:n
    for j = (i+1):n
        if W(i,j) ~= 0
            W1 = W;
            a1 = sum(sum(Wm(1:(i-1),:)));
            a2 = sum(Wm(i,1:j));
            a = a1+a2;
            C(a) = k;
            W(i,j)=0;
            W(i,:)=0;W1(j,:)=0;
            W(:,i)=0;W1(:,j)=0;
            while sum(sum(W1))
                [k1 k2]=find(W1~=0);
                k1 = k1(1);
                k2 = k2(1);
                a1 = sum(sum(Wm(1:(k1-1),:)));
                a2 = sum(Wm(k1,1:k2));
                a = a1+a2;
                C(a)=k;
                W(k1,k2)=0;
                W1(k1,:)=0;W1(k2,:)=0;
                W1(:,k1)=0;W1(:,k2)=0;
            end
            k = k+1;
        end
    end
end
k = k- 1;

end

