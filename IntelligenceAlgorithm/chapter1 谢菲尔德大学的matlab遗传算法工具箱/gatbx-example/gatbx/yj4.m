clc, clear;
%% �ջ�ϵͳ���ſ���
%�����Ŵ��㷨����
NVAR=20;               %����ά��
RANGE=[0;200];         %������Χ
GGAP=0.8;              %����(Generation gap)
XOVR=1;                %������
MUTR=1/NVAR;           %������
MAXGEN=500;            %����Ŵ�����(Maximum number of generations)
INSR=0.9;              %������
SUBPOP=8;              %����Ⱥ��
MIGR=0.2;              %Ǩ����
MIGGEN=20;             %������Ⱥ��Ǩ��֮��20��
NIND=20;               %������Ŀ(Number of individuals)
SEL_F='sus';           %ѡ������
XOV_F='recdis';        %���麯����
MUT_F='mutbga';        %���캯����
OBJ_F='objharv';       %Ŀ�꺯����
FieldDD=rep(RANGE,[1,NVAR]);                         %�������
gen=0;
trace=zeros(MAXGEN,2);                               %�Ŵ��㷨���ܸ���
Chrom=crtrp(SUBPOP*NIND,FieldDD);                    %������ʼ��Ⱥ
ObjV=objharv(Chrom);                                 %����Ŀ�꺯��ֵ
while gen<MAXGEN                                     %��ѭ��
    FitnV=ranking(ObjV,[2 1],SUBPOP);                %������Ӧ��ֵ(Assign fitness values)
    SelCh=select(SEL_F,Chrom,FitnV,GGAP,SUBPOP);                     %ѡ��
    SelCh=recombin(XOV_F,SelCh,XOVR,SUBPOP);                         %����
    SelCh=mutate(MUT_F,SelCh,FieldDD,[MUTR],SUBPOP);                 %����
    ObjVOff=feval(OBJ_F,SelCh);                                      %����Ŀ�꺯��ֵ
    [Chrom, ObjV]=reins(Chrom,SelCh,SUBPOP,[1 INSR],ObjV,ObjVOff);   %���
    gen=gen+1;
    [trace(gen,1),I]=min(ObjV);
    trace(gen,2)=mean(ObjV);
    %������Ⱥ֮��Ǩ�Ƹ���
    if(rem(gen,MIGGEN)==0)
        [Chrom, ObjV]=migrate(Chrom,SUBPOP,[MIGR, 1, 1],ObjV);
    end
end
[Y,I]=min(ObjV);           %������Ž⼰����ţ�YΪ���Ž⣬IΪ��Ⱥ�����
figure(1);plot(Chrom(I,:));
hold on;grid;
plot(Chrom(I,:),'bo')
figure(2);plot(-trace(:,1));
hold on;
plot(-trace(:,2),'-.');
legend('��ı仯','��Ⱥ��ֵ�ı仯');
xlabel('��������')