function D = shortdf(W)
%��ͨͼ�и��������̾���ļ���
% Floyd�㷨
%arg:�ڽӾ��󣬶���W(i,j)�����������ڻ�����Ϊ����Ȩֵ
%����Ϊinf����i=jʱ��W(i,j)=0
n = length(W);
D=W;
m=1;
while m<= n
    for i = 1:n
        for j = 1:n
            if D(i,j) > D(i,m)+D(m,j)
                D(i,j)=D(i,m)+D(m,j);
            end
        end
    end
    m=m+1;
end
D;

end

