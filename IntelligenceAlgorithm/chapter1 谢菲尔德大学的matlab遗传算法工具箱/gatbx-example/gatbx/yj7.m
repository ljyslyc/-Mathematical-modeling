clc, clear;
%% m���ؿյ���������Ԫ��n����ϮĿ�����Ŀ�����
%�����Ŵ��㷨����
NIND=40;                    %������Ŀ(Number of individuals)
MAXGEN=400;                 %����Ŵ�����(Maximum number of generations)
GGAP=0.9;                   %����(Generation gap)
trace=zeros(MAXGEN,2);      %�Ŵ��㷨���ܸ��ٳ�ʼֵ
BaseV=crtbase(15,8);
Chrom=crtbp(NIND, BaseV)+ones(NIND,15);    %��ʼ��Ⱥ
gen=0;
ObjV=targetalloc(Chrom);                   %�����ʼ��Ⱥ����ֵ
while gen<MAXGEN
    FitnV=ranking(-ObjV);                  %������Ӧ��ֵ(Assign fitness values)
    SelCh=select('sus',Chrom,FitnV,GGAP);               %ѡ��
    SelCh=recombin('xovsp',SelCh,0.7);                  %����
    f=rep([1;8],[1,15]);
    SelCh=mutbga(SelCh, f);SelCh=fix(SelCh);            %����
    ObjVSel=targetalloc(SelCh);                         %�����Ӵ�Ŀ�꺯��ֵ
    [Chrom ObjV]=reins(Chrom,SelCh,1,1,ObjV,ObjVSel);   %�ز���
    gen=gen+1;
    trace(gen,1)=max(ObjV);                             %�Ŵ��㷨���ܸ���
    trace(gen,2)=sum(ObjV)/length(ObjV);
end
[Y, I]=max(ObjV);Chrom(I,:),Y                           %���Ž⼰��Ŀ�꺯��ֵ
plot(trace(:,1),'-.');hold on;
plot(trace(:,2));grid
legend('��ı仯','��Ⱥ��ֵ�ı仯')