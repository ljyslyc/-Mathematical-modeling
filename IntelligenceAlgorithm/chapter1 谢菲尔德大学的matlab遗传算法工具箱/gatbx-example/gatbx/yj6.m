clc, clear;
%% ��ɢ��������ϵͳ���ſ�������
%�����Ŵ��㷨����
GGAP=0.8;              %����(Generation gap)
XOVR=1;                %������
NVAR=20;               %����ά��
MUTR=1/NVAR;           %������
MAXGEN=2000;           %����Ŵ�����(Maximum number of generations)
INSR=0.9;              %������
SUBPOP=12;             %�Ӵ���Ŀ
MIGR=0.2;              %Ǩ����
MIGGEN=20;             %ÿ20��Ǩ�Ƹ���
NIND=20;               %������Ŀ(Number of individuals)
SEL_F='sus';           %ѡ������
XOV_F='recdis';        %���麯����
MUT_F='mutbga';        %���캯����
OBJ_F='objlinq';       %Ŀ�꺯����
FieldDR=feval(OBJ_F,[],1);                         
%Chrom=crtrp(SUBPOP*NIND,FieldDR);                    %������ʼ��Ⱥ
Chrom=crtrp(SUBPOP*NIND,FieldDR);
gen=0;
trace=zeros(MAXGEN,2);                               %�Ŵ��㷨���ܸ���
ObjV=feval(OBJ_F,Chrom);                             %����Ŀ�꺯��ֵ
while gen<MAXGEN                                     %��ѭ��
    trace(gen+1,1)=min(ObjV);
    trace(gen+1,2)=mean(ObjV);
    FitnV=ranking(ObjV,[2,0],SUBPOP);                %������Ӧ��ֵ(Assign fitness values)
    SelCh=select(SEL_F,Chrom,FitnV,GGAP,SUBPOP);                     %ѡ��
    SelCh=recombin(XOV_F,SelCh,XOVR,SUBPOP);                         %����
    SelCh=mutate(MUT_F,SelCh,FieldDR,[MUTR],SUBPOP);                 %����
    ObjVOff=feval(OBJ_F,SelCh);                                      %�����Ӵ�Ŀ�꺯��ֵ
    [Chrom, ObjV]=reins(Chrom,SelCh,SUBPOP,[1 INSR],ObjV,ObjVOff);   %���
    gen=gen+1;
    %������Ⱥ֮��Ǩ�Ƹ���
    if(rem(gen,MIGGEN)==0)
        [Chrom, ObjV]=migrate(Chrom,SUBPOP,[MIGR, 1, 1],ObjV);
    end
end
[Y,I]=min(ObjV);                      
subplot(211);
plot(Chrom(I,:));
hold on;
plot(Chrom(I,:),'.');grid             %���ſ��������ֲ�ͼ
legend('���ſ�������')
subplot(212);
plot(trace(:,1));hold on;
plot(trace(:,2),'-.');grid
legend('��ı仯','��Ⱥ��ֵ�ı仯');