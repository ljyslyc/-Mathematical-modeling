close all; clear all; clc;					%�ر�����ͼ�δ��ڣ���������ռ����б��������������
try									%��һ���ļ���Ϊgirl.bmp���ļ������ļ������ڣ����
picture=imread('girl.bmp','bmp');		%һ���ļ���Ϊgirl.jpg���ļ�
filename='girl.bmp';
catch
picture=imread('girl.jpg','jpg');
filename='girl.jpg';
end
filename;
lasterror;
