clc, clear
%% һԪ���庯���Ż�ʵ��
%�����Ŵ��㷨����
NIND=40;               %������Ŀ(Numbe of individuals)
MAXGEN=500;            %����Ŵ�����(Maximum number of generations)
NVAR=20;               %������ά��
PRECI=20;              %�����Ķ�����λ��(Precision of variables)
GGAP=0.9;              %����(Generation gap)
trace=zeros(MAXGEN, 2);
%��������������(Build field descriptor)
FieldD=[rep([PRECI],[1,NVAR]);rep([-512;512],[1, NVAR]);rep([1;0;1;1],[1,NVAR])];
Chrom=crtbp(NIND, NVAR*PRECI);                       %������ʼ��Ⱥ
gen=0;                                               %��������
ObjV=objfun1(bs2rv(Chrom, FieldD));                  %�����ʼ��Ⱥ�����Ŀ�꺯��ֵ
while gen<MAXGEN                                     %����
    FitnV=ranking(ObjV);                             %������Ӧ��ֵ(Assign fitness values)
    SelCh=select('sus', Chrom, FitnV, GGAP);         %ѡ��
    SelCh=recombin('xovsp', SelCh, 0.7);             %����
    SelCh=mut(SelCh);                                %����
    ObjVSel=objfun1(bs2rv(SelCh, FieldD));           %�����Ӵ�Ŀ�꺯��ֵ 
    [Chrom ObjV]=reins(Chrom, SelCh, 1, 1, ObjV, ObjVSel);     %�ز���
    gen=gen+1;                                                 %������������
    trace(gen, 1)=min(ObjV);                                   %�Ŵ��㷨���ܸ���
    trace(gen, 2)=sum(ObjV)/length(ObjV);
end
plot(trace(:,1));hold on;
plot(trace(:,2),'-.');grid;
legend(' ��Ⱥ��ֵ�ı仯','��ı仯')
%������Ž⼰���Ӧ��20���Ա�����ʮ����ֵ,YΪ���Ž�,IΪ��Ⱥ�����
[Y, I]=min(ObjV)
X=bs2rv(Chrom, FieldD);
X(I,:)