function f=target(T,M)                     %��Ӧ�Ⱥ���,TΪ������ͼ��,MΪ��ֵ����
[U, V]=size(T);
W=length(M);
f=zeros(W,1);
for k=1:W
    I=0;s1=0;J=0;s2=0;                     %ͳ��Ŀ��ͼ��ͱ���ͼ���������������֮��
    for i=1:U
        for j=1:V
            if T(i,j)<=M(k)
                s1=s1+T(i,j);I=I+1;
            end
            if T(i,j)>M(k)
                s2=s2+T(i,j);J=J+1;
            end
        end
    end
    if I==0,  p1=0;  else p1=s1/I; end
    if J==0,  p2=0;  else p2=s2/J; end
    f(k)=I*J*(p1-p2)*(p1-p2)/(256*256);
end
