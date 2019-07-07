clc, clear;
%% ��һԪ�����Ż�ʵ��
figure(1);
fplot('variable.*sin(10*pi*variable)+2.0',[-1,2]);   %������������
%�����Ŵ��㷨����
NIND=40;        %������Ŀ(Number of individuals)
MAXGEN=25;      %����Ŵ�����(Maximum number of generations)
PRECI=20;       %�����Ķ�����λ��(Precision of variables)
GGAP=0.9;       %����(Generation gap)
trace=zeros(2, MAXGEN);                        %Ѱ�Ž���ĳ�ʼֵ
FieldD=[20;-1;2;1;0;1;1];                      %����������(Build field descriptor)
Chrom=crtbp(NIND, PRECI);                      %��ʼ��Ⱥ
gen=0;                                         %��������
variable=bs2rv(Chrom, FieldD);                 %�����ʼ��Ⱥ��ʮ����ת��
ObjV=variable.*sin(10*pi*variable)+2.0;        %����Ŀ�꺯��ֵ

while gen<MAXGEN
   FitnV=ranking(-ObjV);                                  %������Ӧ��ֵ(Assign fitness values)         
   SelCh=select('sus', Chrom, FitnV, GGAP);               %ѡ��
   SelCh=recombin('xovsp', SelCh, 0.7);                   %����
   SelCh=mut(SelCh);                                      %����
   variable=bs2rv(SelCh, FieldD);                         %�Ӵ������ʮ����ת��
   ObjVSel=variable.*sin(10*pi*variable)+2.0;             %�����Ӵ���Ŀ�꺯��ֵ
   [Chrom ObjV]=reins(Chrom, SelCh, 1, 1, ObjV, ObjVSel); %�ز����Ӵ�������Ⱥ
   variable=bs2rv(Chrom, FieldD);
   gen=gen+1;                                             %������������
   %������Ž⼰����ţ�����Ŀ�꺯��ͼ���б����YΪ���Ž�,IΪ��Ⱥ�����
   [Y, I]=max(ObjV);hold on;
   plot(variable(I), Y, 'bo');
   trace(1, gen)=max(ObjV);                               %�Ŵ��㷨���ܸ���
   trace(2, gen)=sum(ObjV)/length(ObjV);
end
variable=bs2rv(Chrom, FieldD);                            %���Ÿ����ʮ����ת��
hold on, grid;
plot(variable,ObjV,'b*');
figure(2);
plot(trace(1,:));
hold on;
plot(trace(2,:),'-.');grid
legend('��ı仯','��Ⱥ��ֵ�ı仯')