%% ��η�����
clc, clear

fid = fopen('data.txt', 'r');
n1 = 6;  %׼���ָ�����
n2 = 3;  %������ָ�����
a = [];
% ��ȡ׼����жϾ���
for i = 1:n1
    tmp = str2num(fgetl(fid));
    a = [a;tmp];
end
% ��ȡ�������жϾ����жϾ������b1~b6��
for i = 1:n1
    str1 = char(['b', int2str(i), '=[];']);
    str2 = char(['b', int2str(i), '=[b', int2str(i), ';tmp];']);
    eval(str1);
    for j = 1:n2
        tmp = str2num(fgetl(fid));
        eval(str2);
    end
end

% ���������ֵ����Ӧ�Ĺ�һ����������
[max(1),wA]=ahp(a);
for i = 1:n1
    str = char(['[max(', int2str(i+1), '),wb', int2str(i), ']=ahp(', 'b', int2str(i), ')']);
    eval(str);
end

%����ƽ��һ����ָ��
RIT=CalculationRI();

% д��ѭ��
[RIA,CIA]=sglsortexamine(max(1),a,RIT);
[RIb1,CIb1]=sglsortexamine(max(2),b1,RIT);
[RIb2,CIb2]=sglsortexamine(max(3),b2,RIT);
[RIb3,CIb3]=sglsortexamine(max(4),b3,RIT);
[RIb4,CIb4]=sglsortexamine(max(5),b4,RIT);
[RIb5,CIb5]=sglsortexamine(max(6),b5,RIT);
[RIb6,CIb6]=sglsortexamine(max(7),b6,RIT);

dw=zeros(3,6);
% д��ѭ��
dw(1:3,1)=wb1;
dw(1:3,2)=wb2;
dw(1:3,3)=wb3;
dw(1:3,4)=wb4;
dw(1:3,5)=wb5;
dw(1:3,6)=wb6;

CIC=[CIb1;CIb2;CIb3;CIb4;CIb5;CIb6];
RIC=[RIb1;RIb2;RIb3;RIb4;RIb5;RIb6];
tw=tolsortvec(wA,dw,CIC,RIC)';
wA %���׼����Ŀ���Ȩ��
dw %���׼���Է�����Ȩ��
tw  %���������Ȩֵ
res=tw';
[diffcult,num]=sort(res);
[diffcult,num]
