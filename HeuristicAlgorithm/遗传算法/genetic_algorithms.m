%��γ����СTSP����
clc,clear

load example_1.txt %���صз�100 ��Ŀ������ݣ��޸ģ�ʹ�䴦��5��Ŀ�꣩
sj = example_1

x=sj(:,1:2:8);x=x(:);%����
y=sj(:,2:2:8);y=y(:);%γ��
sj=[x y];
d1=[70,40];
sj0=[d1;sj;d1];

%�������d
sj=sj0*pi/180;
d=zeros(102);
for i=1:101
    for j=i+1:102%����һ�������Ǿ���
        temp=cos(sj(i,1)-sj(j,1))*cos(sj(i,2))*cos(sj(j,2))+sin(sj(i,2))*sin(sj(j,2));
        d(i,j)=6370*acos(temp);
    end
end

d=d+d';L=102;w=50;dai=100;%w��Ⱥ��С��dai����
%ͨ������Ȧ�㷨ѡȡ��������A
for k=1:w
    c=randperm(100);%����0~100�������
    c1=[1,c+1,102];
    flag=1;
    while flag>0
        flag=0;
        for m=1:L-3
            for n=m+2:L-1
                if d(c1(m),c1(n))+d(c1(m+1),c1(n+1))<d(c1(m),c1(m+1))+d(c1(n),c1(n+1))%�ж���·����ԭ·����Ƚ�
                    flag=1;
                    c1(m+1:n)=c1(n:-1:m+1);%cl(2:4)=cl(4:-1:2)��ת
                end
            end
        end
    end
    J(k,c1)=1:102;%k=1,2,3��
end

J=J/102;%��һ��
J(:,1)=0;J(:,102)=1;
rand('state',sum(clock));

%�Ŵ��㷨ʵ�ֹ���
A=J;
for k=1:dai %����0��1 ��������н��б���
    B=A;
    c=randperm(w);%w��Ⱥ��С
    %��������Ӵ�B
    for i=1:2:w
        F=2+floor(100*rand(1));%rand(1)����һ���������0,1��floor��������ȡ��
        temp=B(c(i),F:102);%B(2,63:102)
        B(c(i),F:102)=B(c(i+1),F:102);%����֮�佻������
        B(c(i+1),F:102)=temp;
    end
    %-280-
    %��������Ӵ�C
    by=find(rand(1,w)<0.1);%������С��0.1
    if length(by)==0
        by=floor(w*rand(1))+1;
    end
    C=A(by,:);
    L3=length(by);
    for j=1:L3
        bw=2+floor(100*rand(1,3));
        bw=sort(bw);%��������
        C(j,:)=C(j,[1:bw(1)-1,bw(2)+1:bw(3),bw(1):bw(2),bw(3)+1:102]);%C��1,[1:16,65:73,17:64,74:102]��
    end
    G=[A;B;C];%105*102
    TL=size(G,1);
    %�ڸ������Ӵ���ѡ������Ʒ����Ϊ�µĸ���
    [dd,IX]=sort(G,2);temp(1:TL)=0;%sort �����ÿһ�а���������
    for j=1:TL
        for i=1:101
            temp(j)=temp(j)+d(IX(j,i),IX(j,i+1));
        end
    end
    [DZ,IZ]=sort(temp);
    A=G(IZ(1:w),:);
end

path=IX(IZ(1),:)
long=DZ(1)
%toc
xx=sj0(path,1);yy=sj0(path,2);%���Ⱥ�γ��
%DD=[xx,yy];
plot(xx,yy,'-o')

% [B,I]=sort(A,2) 
%B=[1 3 5
%   1 2 4]
%I=[ 1 3 2   Ԫ����ԭ�������е�λ��
%    3 1 2]

%[B,I]=sort(A) I����Ԫ����ԭ�������е�λ��
