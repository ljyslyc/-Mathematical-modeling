function [BW,runningt]=Denoise(RGB,M)
 %RGBԭͼ��M��ʾ���������Ĵ�����BWΪ����������ͼ��runningtΪ��������ʱ��

A=imnoise(RGB,'gaussian',0,0.05);   %�����˹������
I=A;                                %��A��ֵ��I
I=im2double(I);                     %��I��������ת����˫����
RGB=im2double(RGB);                     
tstart=tic; %��ʼ��ʱ
for i=1:M
   I=imadd(I,RGB);                  %����ԭͼ���������ͼ����ж�ε��ӣ�������ظ�I
end
avg_A=I/(M+1);                      %����ӵ�ƽ��ͼ�� 
runningt=toc(tstart);               %��ʱ����
BW=avg_A;