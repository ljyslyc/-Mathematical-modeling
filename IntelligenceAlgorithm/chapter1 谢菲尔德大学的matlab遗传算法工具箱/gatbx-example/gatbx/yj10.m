clc, clear;
%% ͼ��ָ�����
% �����Ŵ��㷨����ͼ��ָ�Ļ���˼���ǣ���ͼ���е����ذ��Ҷ�ֵ����ֵM�ֳ�����ͼ��
% һ��ΪĿ��ͼ�� ����һ��Ϊ����ͼ�� ��
% ͼ�� �ɻҶ�ֵ��0��M֮���������ɣ�ͼ�� �ɻҶ�ֵ��M+1��L-1��LΪͼ��ĻҶȼ�����֮���������ɡ�
% ��������ͼ��Ϊ256�Ҷȼ������Ҷȷָ���ֵ����Ϊһ��8λ0��1�������봮
load woman                      %����MATLAB��Womanͼ��Ҷ�ֵ
figure(1);                      %��ͼ
image(X);colormap(map);         
NIND=40;                        %������Ŀ(Number of individuals)
MAXGEN=50;                      %����Ŵ�����(Maximum number of generations)
PRECI=8;                        %�����Ķ�����λ��(Precision of variables)
GGAP=0.9;                       %����(Generation gap)
FieldD=[8;1;256;1;0;1;1];       %��������������(Build field descriptor)
Chrom=crtbp(NIND,PRECI);        %������ʼ��Ⱥ
gen=0;    
phen=bs2rv(Chrom,FieldD);       %��ʼ��Ⱥʮ����ת��
ObjV=target(X,phen);            %������Ⱥ��Ӧ��ֵ
while gen<MAXGEN                %����(Generation gap)
    FitnV=ranking(-ObjV);       %������Ӧ��ֵ(Assign fitness values)
    SelCh=select('sus',Chrom,FitnV,GGAP);     %ѡ��
    SelCh=recombin('xovsp',SelCh,0.7);        %����
    SelCh=mut(SelCh);                         %����
    phenSel=bs2rv(SelCh,FieldD);              %�Ӵ�ʮ����ת��
    ObjVSel=target(X,phenSel);
    [Chrom ObjV]=reins(Chrom,SelCh,1,1,ObjV,ObjVSel);  %�ز���
    gen=gen+1;
end
[Y, I]=max(ObjV);
M=bs2rv(Chrom(I,:),FieldD);                   %������ֵ
[m, n]=size(X);
for i=1:m
    for j=1:n
        if X(i,j)>M                           %�Ҷ�ֵ������ֵʱ�ǰ�ɫ
            X(i,j)=256;
        end
    end
end
figure(2)                                     %�����ָ��Ŀ��ͼ��
image(X);colormap(map);