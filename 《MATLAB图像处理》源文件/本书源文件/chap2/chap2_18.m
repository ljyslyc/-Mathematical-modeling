close all; clear all; clc;						%�ر�����ͼ�δ��ڣ���������ռ����б��������������
stu_cell={'LiMing','20120101','M','20'};			%����ϸ������
celldisp(stu_cell)							%��ʾϸ������
fields={'name','number','sex','age'};
stu_struct=cell2struct(stu_cell,fields,2);			%��ϸ������ת���ɽṹ��
stu_struct;
a=iscell(stu_cell);							%�ж�stu_cell�Ƿ�Ϊϸ������
b=iscell(stu_struct);
stu_t=struct('name',{'LiMing','WangHong'},'number',{'20120101','20120102'},'sex',{'f','m' },'age',{20,19});
stu_c=struct2cell(stu_t);						%���ṹ��ת����ϸ������
c= {[1] [2 3 4]; [5; 9] [6 7 8; 10 11 12]};			%����ϸ������
m= cell2mat(c);								%��ϸ������ϲ��ɾ���
M = [1 2 3 4 5; 6 7 8 9 10; 11 12 13 14 15; 16 17 18 19 20];
C1= mat2cell(M, [2 2], [3 2]);					%��M��ֳ�ϸ������
C2=num2cell(M);							%��Mת����ϸ������
figure;
subplot(121);cellplot(C1);						%��ʾC1�ṹͼ
subplot(122);cellplot(C2);						%��ʾC2�ṹͼ
