close all; clear all; clc;					%�ر�����ͼ�δ��ڣ���������ռ����б��������������
stu=struct('name',{'LiMing','WangHong'},'number',{'20120101','20120102'},'sex',{'f','m' },'age',{20,19});
a=fieldnames(stu);                      	%��ȡstu���г�Ա��
b=getfield(stu,{1,2},'name');				%��ȡָ����Ա����
c=isfield(stu,'sex');						%�ж�sex�Ƿ�Ϊstu�г�Ա
stunew=orderfields(stu);					%���ṹ���Ա����ĸ��������
rmfield(stu,'sex');                         	%ɾ��sex
s1=setfield(stu(1,1),'sex','M');				%��������stu��sex����
s2=setfield(stu{1,2},'sex','F'); 				%��������stu��sex����
s2(1,2)
