NIND=100;                              %������Ŀ(Number of individuals)
MAXGEN=50;                             %����Ŵ�����(Maximum number of generations)
NVAR=2;                                %��������
PRECI=20;                              %�����Ķ�����λ��(Precision of variables)
GGAP=0.9;                              %����(Generation gap)
trace1=[];trace2=[];trace3=[];         %���ܸ���
%��������������(Build field descriptor)
FieldD=[rep([PRECI],[1,NVAR]);[1,1;4,2];rep([1;0;1;1],[1,NVAR])];
Chrom=crtbp(NIND,NVAR*PRECI);          %��ʼ��Ⱥ
v=bs2rv(Chrom,FieldD);                 %��ʼ��Ⱥʮ����ת��
gen=1;
while gen<MAXGEN
    ObjV1=f1(v(1:NIND/2,:));           %������һĿ�꺯��ֵ
    FitnV1=ranking(ObjV1);             %������Ӧ��ֵ(Assign fitness values)
    SelCh1=select('sus',Chrom(1:NIND/2,:),FitnV1,GGAP);                 %ѡ��
    ObjV2=f2(v(NIND/2+1:NIND,:));      %�����ڶ�Ŀ�꺯��ֵ
    FitnV2=ranking(ObjV2);
    SelCh2=select('sus',Chrom(NIND/2+1:NIND,:),FitnV2,GGAP);            %ѡ��
    ObjV=[ObjV1;ObjV2];
    SelCh=[SelCh1;SelCh2];             %�ϲ�
    SelCh=recombin('xovsp',SelCh,0.7); %����
    SelCh=mut(SelCh);
    v=bs2rv(SelCh,FieldD);
    ObjVSel1=f1(v(1:NIND/2*GGAP,:));
    ObjVSel2=f2(v(NIND/2*GGAP+1:NIND*GGAP,:));
    ObjVSel=[ObjVSel1;ObjVSel2];       %�����Ӵ�������Ŀ�꺯��ֵ��ϲ���һ��
    [Chrom ObjV]=reins(Chrom,SelCh,1,1,ObjV,ObjVSel);                    %�ز����Ӵ�������Ⱥ
    v=bs2rv(Chrom,FieldD);
    
    trace1(gen,1)=min(f1(v));
    trace1(gen,2)=sum(f1(v))/length(f1(v));
    trace2(gen,1)=min(f2(v));
    trace2(gen,2)=sum(f2(v))/length(f2(v));
    trace3(gen,1)=min(f1(v)+f2(v));
    trace3(gen,2)=sum(f1(v))/length(f1(v))+sum(f2(v))/length(f2(v));
    gen=gen+1;
end
figure(1);clf;
plot(trace1(:,1));hold on;plot(trace1(:,2),'-.');
plot(trace1(:,1),'.');plot(trace1(:,2),'.');grid;
legend('��ı仯','��Ⱥ��ֵ�ı仯')
xlabel('��������');ylabel('��һĿ�꺯��ֵ');
figure(2);clf;
plot(trace2(:,1));hold on;
plot(trace2(:,2),'-.');
plot(trace2(:,1),'.');
plot(trace2(:,2),'.');grid;
legend('��ı仯','��Ⱥ��ֵ�ı仯')
xlabel('��������');ylabel('�ڶ�Ŀ�꺯��ֵ');
figure(3);clf;
plot(trace3(:,1));hold on;
plot(trace3(:,2),'-.');
plot(trace3(:,1),'.');
plot(trace3(:,2),'.');grid;
legend('��ı仯','��Ⱥ��ֵ�ı仯')
xlabel('��������');ylabel('Ŀ�꺯��ֵ֮��');
figure(4);clf;plot(f1(v));hold on;
plot(f2(v),'r-.');grid;