close all;                   %�رյ�ǰ����ͼ�δ��ڣ���չ����ռ��������������ռ����б���
clear all;
clc
A=imread('tire.tif');        %��ȡͼ��tire������ֵ��A
[m,n]=size(A);               %��ȡͼ�����A��������m��n
B=imread('eight.tif');      %��ȡͼ��eight��ֵ������ֵ��B
C=B;                        %��ʼ������C
A=im2double(A);           %����A\B\C����������Ϊ˫����  
B=im2double(B);
C=im2double(C);
for i=1:m                      %��ͼ��B��A���ӣ������ֵ��C
    for j=1:n
    C(i,j)=B(i,j)+A(i,j);   
    end
end
D=imabsdiff(C,B);           %����Ӻ�ͼ��C��B�Ĳ��죬��ֵ��D     
set(0,'defaultFigurePosition',[100,100,1000,500]);%�޸�ͼ��ͼ��λ�õ�Ĭ������
set(0,'defaultFigureColor',[1 1 1])%�޸�ͼ�α�����ɫ������
figure(1),
subplot(121),imshow(A); %��ʾtire��eightͼ�񣬵���ͼ�񼰲���ͼ��
subplot(122),imshow(B);
figure(2),
subplot(121),imshow(C);
subplot(122),imshow(D);
