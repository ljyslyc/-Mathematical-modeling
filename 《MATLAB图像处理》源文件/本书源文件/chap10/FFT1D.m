function B1=FFT1D(A1)
%%��������A1�е����ݽ�����ż�ֽ�����-----
B=SortOE(A1);
n=length(B);m=log2(n);
for s=1:n
    T(s)=double(B(s));%��ͼ������ת��Ϊdouble��
end

for a=0:m-1
    M=2^a;nb=n/M/2;%ÿһ��İ볤�ȺͷֳɵĿ���
    for j=0:nb-1 %��ÿһ�����ν��в���
        for k=0:M-1%��ÿһ���е�һ��ĵ����β���
            t1=double(T(1+k+j*2*M));t2=double(T(1+k+j*2*M+M))*exp(-i*pi*k/M);
            T(1+k+j*2*M)=0.5*(t1+t2);
            T(1+k+j*2*M+M)=0.5*(t1-t2);
        end
    end
end
B1=T;
