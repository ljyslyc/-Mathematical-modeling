clc, clear;
%% ��Ԫ��庯�����Ż�ʵ��:shubert����
[x1,x2]=meshgrid(-10:.1:10);
figure(1);mesh(x1,x2,shubert(x1,x2));            %����shubert����ͼ��
%�����Ŵ��㷨����
NIND=40;               %������Ŀ(Number of individuals)
MAXGEN=50;             %����Ŵ�����(Maximum number of generations)
NVAR=2;                %������Ŀ
PRECI=25;              %�����Ķ�����λ��(Precision of variables)
GGAP=0.9;              %����(Generation gap)
%��������������(Build field descriptor)
FieldD=[rep([PRECI],[1,NVAR]);rep([-10;10],[1,NVAR]);rep([1;0;1;1],[1,NVAR])];
Chrom=crtbp(NIND, NVAR*PRECI);                         %������ʼ��Ⱥ
gen=0;                                                 
trace=zeros(MAXGEN, 2);                                %�Ŵ��㷨���ܸ��ٳ�ʼֵ
x=bs2rv(Chrom, FieldD);                                %��ʼ��Ⱥʮ����ת��
ObjV=shubert(x(:,1),x(:,2));                           %�����ʼ��Ⱥ��Ŀ�꺯��ֵ
while gen<MAXGEN
    FitnV=ranking(ObjV);                               %������Ӧ��ֵ(Assign fitness values)
    SelCh=select('sus',Chrom,FitnV,GGAP);              %ѡ��
    SelCh=recombin('xovsp',SelCh,0.7);                 %����
    SelCh=mut(SelCh);                                  %����
    x=bs2rv(SelCh,FieldD);                             %�Ӵ�ʮ����ת��
    ObjVSel=shubert(x(:,1),x(:,2));
    [Chrom ObjV]=reins(Chrom,SelCh,1,1,ObjV,ObjVSel);  %�ز���
    gen=gen+1;
    [Y, I]=min(ObjV);
    Y,bs2rv(Chrom(I,:),FieldD)                         %���ÿһ�ε����Ž⼰���Ӧ���Ա���ֵ
    trace(gen,1)=min(ObjV);                            %�Ŵ��㷨���ܸ���
    trace(gen,2)=sum(ObjV)/length(ObjV);
    if(gen==50)                                        %������Ϊ50ʱ����Ŀ�꺯��ֵ�ֲ�ͼ
        figure(2);
        plot(ObjV);hold on;
        plot(ObjV,'b*');grid;
    end
end
figure(3);clf;
plot(trace(:,1));hold on;
plot(trace(:,2),'-.');grid
legend('��ı仯','��Ⱥ��ֵ�ı仯')