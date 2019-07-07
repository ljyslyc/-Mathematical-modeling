clc, clear;
%% ˫���ֵ��Ż�����
%�����Ŵ��㷨����
Dim=20;               %����ά��
NIND=20;              %������Ŀ(Number of individuals)
Preci=20;             %�����Ķ�����λ��(Precision of variables)
MAXGEN=100;            %����Ŵ�����(Maximum number of generations)
GGAP=0.8;             %����(Generation gap)
SEL_F='sus';          %ѡ������
XOV_F='xovsp';        %���麯����
MUT_F='mut';          %���캯����
OBJ_F='objdopi';      %Ŀ�꺯����
FieldDR=feval(OBJ_F,[],1);                    %����Ŀ�꺯��ֵ
%��������������(Build field descriptor)
FieldDD=[rep([Preci],[1,Dim]);FieldDR;rep([1;0;1;1],[1,Dim])];
Chrom=crtbp(NIND, Dim*Preci);                 %������ʼ��Ⱥ
gen=0;
Best=NaN*ones(MAXGEN,1);                      %���Ž��ֵ
while gen<MAXGEN                              %���ѭ������
    ObjV=feval(OBJ_F,bs2rv(Chrom,FieldDD));   %����Ŀ�꺯��ֵ
    Best(gen+1)=min(ObjV);                    %���Ž�
    plot(log10(Best),'bo');
    FitnV=ranking(ObjV);                      %������Ӧ��ֵ(Assign fitness values)
    SelCh=select(SEL_F,Chrom,FitnV,GGAP);     %ѡ��
    SelCh=recombin(XOV_F,SelCh);              %����
    SelCh=mutate(MUT_F,SelCh);                %����
    Chrom=reins(Chrom,SelCh);                 %�ز���
    gen=gen+1;
end
grid;
xlabel('��������');ylabel('Ŀ�꺯��ֵ(ȡ����)');