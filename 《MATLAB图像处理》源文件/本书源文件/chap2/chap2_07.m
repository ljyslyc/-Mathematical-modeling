close all; clear all; clc;				%�ر�����ͼ�δ��ڣ���������ռ����б��������������
S1='Good morning!';
S2='good morning, Sir.';
a=strcmp(S1,S2);					%�Ƚ������ַ�����С
b=strncmp(S1,S2,7);					%�Ƚ������ַ���ǰ7���ַ���С�����ִ�Сд
c=strncmpi(S1,S2,7);				%�Ƚ������ַ���ǰ7���ַ���С�������ִ�Сд
