%��ż�ֽ�������
function  B=SortOE(T)
    n=length(T);m=log2(n/2);
for i=1:m
    nb=2^i;lb=n/nb;%�ֳɵĿ�����ÿһ��ĳ���
    lc=2*lb;%�������
        for j=0:nb/2-1   %������������Ĵ���
            t=T(2+j*lc:2:2*lb+j*lc);
            T(1+j*lc:lb+j*lc)=T(1+j*lc:2:(2*lb-1)+j*lc);
            T(lb+1+j*lc:2*lb+j*lc)=t;
        end
 end
B=T;
