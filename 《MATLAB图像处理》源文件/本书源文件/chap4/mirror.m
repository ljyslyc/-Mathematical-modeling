function OutImage=mirror(InImage,n)
%mirror����ʵ��ͼ����任����
%����nΪ1ʱ��ʵ��ˮƽ����任
%����nΪ2ʱ��ʵ�ִ�ֱ����任
%����nΪ3ʱ��ʵ��ˮƽ��ֱ����任
I=InImage;
[M,N,G]=size(I);%��ȡ����ͼ��I�Ĵ�С
J=I;  %��ʼ����ͼ�����ȫΪ1����С������ͼ����
if (n==1)
    for i=1:M
        for j=1:N
            J(i,j,:)=I(M-i+1,j,:);%n=1,ˮƽ����
        end
    end;
elseif (n==2)
     for i=1:M
        for j=1:N
            J(i,j,:)=I(i,N-j+1,:);%n=2,��ֱ����
        end
     end    
elseif (n==3)
     for i=1:M
        for j=1:N
            J(i,j,:)=I(M-i+1,N-j+1,:);%n=3,ˮƽ��ֱ����
        end
     end
else
    error('����n���벻��ȷ��nȡֵ1��2��3')%n�������ʱ��ʾ
end
    OutImage=J;

    