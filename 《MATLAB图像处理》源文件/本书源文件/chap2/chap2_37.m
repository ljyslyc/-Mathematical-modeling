close all; clear all; clc;					%�ر�����ͼ�δ��ڣ���������ռ����б��������������
A=magic(5);							%����5��ħ������A
a=A(1);								%����A�����Ԫ�أ���Ԫ���±�
for i=2:25
    if A(i)>a
        a=A(i);
        n=i;
    end
end
