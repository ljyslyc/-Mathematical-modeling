close all; clear all; clc;					%�ر�����ͼ�δ��ڣ���������ռ����б��������������
f1=@help;								%�����������
s1=func2str(f1);							%���������ת�����ַ���
f2=str2func('help');						%���ַ���ת���ɺ������
a1=isa(f1,'function_handle');				%�ж�f1�Ƿ�Ϊ�������
a2=isequal(f1,f2);						%�ж�f1��f2�Ƿ�ָ��ͬһ����
a3=functions(f1);						%��ȡf1��Ϣ
