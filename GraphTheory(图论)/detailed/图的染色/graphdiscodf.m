function [k C W] = graphdiscodf(M)

% ͼ���ڵ������Ⱦɫ����
% M��ʾ����ͼ���ڽӾ���
% k��ʾȾɫ��
% C��ʾ����Ⱦɫ����
% W��ʾ�߼��ϵ�Ⱦɫ����

[k C W]=graphcodf(M);
n = size(M,1);
C1 = [C' W];
k1 = max(max(C1))+1;
k2 = k1;
for i = 1:n
    for j = (i+1):n
        tms = setdiff(C1(i,:),C1(j,:));
        if isempty(tms)
            C1(j,1)= k1;
            k1 = k1+1;
        end
    end
end
k = k+k1-k2;
C = C1(:,1)';
W = C1(:,2:(n+1));
end

