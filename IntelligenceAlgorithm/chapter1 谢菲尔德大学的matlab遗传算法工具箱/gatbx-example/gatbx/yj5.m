clc, clear;
%% װ��ϵͳ����������
%�����Ŵ��㷨����
GGAP=0.8;              %����(Generation gap)
XOVR=1;                %������
NVAR=20;               %����ά��
MUTR=1/NVAR;           %������
MAXGEN=200;            %����Ŵ�����(Maximum number of generations)
INSR=0.9;              %������
SUBPOP=12;             %����Ⱥ��
MIGR=0.2;              %Ǩ����
MIGGEN=20;             %ÿ20��Ǩ�Ƹ���
NIND=20;               %������Ŀ(Number of individuals)
RANGE=[0;10];          %������Χ
SEL_F='sus';           %ѡ������
XOV_F='recdis';        %���麯����
MUT_F='mutbga';        %���캯����
OBJ_F='objpush';       %Ŀ�꺯����
FieldDD=rep(RANGE,[1,NVAR]);                         
trace=zeros(MAXGEN,2);                               %�Ŵ��㷨���ܸ���
Chrom=crtrp(SUBPOP*NIND,FieldDD);                    %������ʼ��Ⱥ
gen=0;
ObjV=feval(OBJ_F,Chrom);                             
while gen<MAXGEN                                     %��ѭ��
    FitnV=ranking(ObjV,[2 0],SUBPOP);                %������Ӧ��ֵ(Assign fitness values)
    SelCh=select(SEL_F,Chrom,FitnV,GGAP,SUBPOP);                     %ѡ��
    SelCh=recombin(XOV_F,SelCh,XOVR,SUBPOP);                         %����
    SelCh=mutate(MUT_F,SelCh,FieldDD,[MUTR],SUBPOP);                 %����
    ObjVOff=feval(OBJ_F,SelCh);                                      %�����Ӵ�Ŀ�꺯��ֵ
    [Chrom, ObjV]=reins(Chrom,SelCh,SUBPOP,[1 INSR],ObjV,ObjVOff);   %���
    gen=gen+1;
    [trace(gen,1),I]=min(ObjV);
    trace(gen,2)=mean(ObjV);
    %������Ⱥ֮��Ǩ�Ƹ���
    if(rem(gen,MIGGEN)==0)
        [Chrom, ObjV]=migrate(Chrom,SUBPOP,[MIGR, 1, 1],ObjV);
    end
end
[Y,I]=min(ObjV);                      %���ſ�������ֵ�������
subplot(211);
plot(Chrom(I,:));hold on;
plot(Chrom(I,:),'.');grid
subplot(212);
plot(trace(:,1));hold on;
plot(trace(:,2),'-.');grid
legend('��ı仯','��Ⱥ��ֵ�ı仯');
xlabel('��������')