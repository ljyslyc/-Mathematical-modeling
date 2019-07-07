%[10.7]
close all; clear all; clc;		%�ر�����ͼ�δ��ڣ���������ռ����б��������������
I=[0 0 1 1 0 0 1 1;1 0 0 1 0 0 1 1;1 1 0 0 0 0 1 0];%������ľ���
[m,n]=size(I);				%��������С
I=double(I);
p_table=tabulate(I(:));	%ͳ�ƾ�����Ԫ�س��ֵĸ��ʣ���һ��Ϊ����Ԫ�أ��ڶ���Ϊ������������Ϊ���ʰٷ���
color=p_table(:,1)';
p=p_table(:,3)'/100;			%ת����С����ʾ�ĸ���
psum=cumsum(p_table(:,3)');	%����������е��ۼ�ֵ
allLow=[0,psum(1:end-1)/100];%���ھ�����Ԫ��ֻ�����֣���[0,1�����仮��Ϊ��������allLow�� allHigh 
allHigh=psum/100;
numberlow=0;				%�������������������numberlow��numberhigh
numberhigh=1;
for k=1:m					%���¼�����������������ޣ���������
   for kk=1:n
       data=I(k,kk);
       low=allLow(data==color);
       high=allHigh(data==color);
       range=numberhigh-numberlow;
       tmp=numberlow;
       numberlow=tmp+range*low;
       numberhigh=tmp+range*high;
   end
end
fprintf('�������뷶Χ����Ϊ%16.15f\n\n',numberlow);
fprintf('�������뷶Χ����Ϊ%16.15f\n\n',numberhigh);
Mat=zeros(m,n);				%����
for k=1:m
   for kk=1:n
       temp=numberlow<low;
       temp=[temp 1];
       indiff=diff(temp);
       indiff=logical(indiff);
       Mat(k,kk)=color(indiff);
       low=low(indiff);
       high=allHigh(indiff);
       range=high - low;
       numberlow=numberlow-low;
       numberlow=numberlow/range;
   end
end
fprintf('ԭ����Ϊ:\n')
disp(I);
fprintf('\n');
fprintf('�������:\n');
disp(Mat);
