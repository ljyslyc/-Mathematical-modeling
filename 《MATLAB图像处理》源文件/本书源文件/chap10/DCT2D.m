function C=DCT2D(B)
%��ͼ�����ݽ��п��ٸ���Ҷ�任�����ط��Ⱥ���λ��Ϣ
a=length(B);
%���ζ�ÿһ�н���FFT����
C=zeros(a);
for b=1:a
    C(b,:)=DCT1D(B(b,:));
end
%���ζ�ÿһ�н���FFT����
for b=1:a
    T=C(:,b);
    T1=DCT1D(T');
    C(:,b)=T1';
end
