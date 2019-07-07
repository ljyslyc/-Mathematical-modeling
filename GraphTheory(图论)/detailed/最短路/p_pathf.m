function [ Path P f ] = p_pathf(A,k1,k2)
%�����ɿ�·���㷨
% Path��ʾ·��,P�Ǹ��ʵļ���ֵ��f=1��ʾ�ҵ�·��f=0��ʾû��·

[m n] = size(A);
f = 1;
B = zeros(m,n);

% ��ԭ�������ת��
for i =1:m
    for j = 1:n
        if A(i,j) > 0 && A(i,j) < 1
            B(i,j) = -log(A(i,j));
        elseif A(i,j)==0
            B(i,j)=inf;
        end
    end
end

%  ����Floyd�㷨�����·
[Path d]=n2shortf(B,k1,k2);
if d < inf
    P=1;
    for i=1:(length(Path)-1)
        P=P*A(Path(i),Path(i+1));
    end
else
    Path = 0;
    P=0;
    f=0;
end



end

