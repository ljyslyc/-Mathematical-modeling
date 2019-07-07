function [ d0 d ] = centgraf(W,A)
%��ͨͼ�����ĺͼ�Ȩ�����㷨
% ��ֻ��ͼ�����ģ�����÷�ʽΪd0=centgraf(W)
% �����߾�������÷�ʽΪ[ d0 d ] = centgraf(W,A),����d0Ϊͼ������
% d�Ǽ�Ȩ����
% arg��W��ʾͼ��Ȩֵ����,A��ʾ�����Ȩ��,d0��ʾͼ������
%      d��ʾ��Ȩ����

% ������̾������
n=length(W);
U = W;
m=1;
while m <= n
    for i = 1:n
        for j = 1:n
            if U(i,j) > U(i,m)+U(m,j)
                U(i,j)= U(i,m)+U(m,j);
            end
        end
    end
    m=m+1;
end

% ������е����ֵ
d1=max(U,[],2);
% �������ֵ�е���Сֵ
d0t=min(d1);
d0=find(d1==d0t);

% �����Ȩ����
dt = zeros(1,n);
for i = 1:n
    dt(i) = dot(U(i,:),A);
end
d = find(dt == min(dt));



end

