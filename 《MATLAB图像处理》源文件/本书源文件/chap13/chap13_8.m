close all;                  %�رյ�ǰ����ͼ�δ��ڣ���չ����ռ��������������ռ����б���
clear all;
clc;
load woman;               %��ȡͼ������
[c,s]=wavedec2(X,2,'db2');%����db4С������2��ͼ��ֽ�
nbcol=size(map,1);
s1=s(1,:);                %��ȡС���ֽ�ϵ�������С
s2=s(3,:);
ca2=zeros(s1);            %��ʼ���ֽ�ϵ������
chd2=zeros(s1);
cvd2=zeros(s1);
cdd2=zeros(s1);
chd1=zeros(s2);
cvd1=zeros(s2);
cdd1=zeros(s2);
l1=s1(1)*s1(1);
l2=s2(1)*s2(1);
%�ӷֽ�ϵ������C�ͳ��Ⱦ���S����ȡϸ��
ca2=reshape(c(1:l1),s1(1),s1(1));%��ȡ��2��С���任�Ľ���ϵ��
chd2=reshape(c(l1+1:2*l1),s1(1),s1(1));%��ȡͼ���2���ϸ��ϵ����ˮƽ����
cvd2=reshape(c(2*l1+1:3*l1),s1(1),s1(1));%��ȡͼ���2���ϸ��ϵ���Ĵ�ֱ����
cdd2=reshape(c(3*l1+1:4*l1),s1(1),s1(1));%��ȡͼ���2���ϸ��ϵ���ĶԽǷ���
chd1=reshape(c(4*l1+1:4*l1+l2),s2(1),s2(1));%��ȡͼ���1���ϸ��ϵ����ˮƽ����
cvd1=reshape(c(4*l1+l2+1:4*l1+2*l2),s2(1),s2(1));%��ȡͼ���1���ϸ��ϵ���Ĵ�ֱ����
cdd1=reshape(c(4*l1+2*l2+1:4*l1+3*l2),s2(1),s2(1));%��ȡͼ���1���ϸ��ϵ���ĶԽǷ���
%���ú���appcoef2()��detcoef2()��ȡС���ֽ�ϵ��
ca2_1=appcoef2(c,s,'db2',2);%��ȡ��2��С���任�Ľ���ϵ��
chd2_1=detcoef2('h',c,s,2); %��ȡͼ���2���ϸ��ϵ����ˮƽ����
cvd2_1=detcoef2('v',c,s,2); %��ȡͼ���2���ϸ��ϵ���Ĵ�ֱ����
cdd2_1=detcoef2('d',c,s,2); %��ȡͼ���2���ϸ��ϵ���ĶԽǷ���
chd1_1=detcoef2('h',c,s,1); %��ȡͼ���1���ϸ��ϵ����ˮƽ����
cvd1_1=detcoef2('v',c,s,1); %��ȡͼ���1���ϸ��ϵ���Ĵ�ֱ����
cdd1_1=detcoef2('d',c,s,1); %��ȡͼ���1���ϸ��ϵ���ĶԽǷ���
disp('�Ƚ����ַ�����ȡС���ֽ�ϵ���Ƿ���ͬ��')
disp(' ')
if isequal(ca2,ca2_1)
    disp('      ca2��ca2_1��ͬ')
    disp(' ')
end
if isequal(chd2,chd2_1)
    disp('      chd2��chd2_1��ͬ')
    disp(' ')
end    
if isequal(cvd2,cvd2_1)
    disp('      cvd2��cvd2_1��ͬ')
    disp(' ')
end
if isequal(cdd2,cdd2_1)
    disp('      cdd2��cdd2_1��ͬ')
    disp(' ')
end    
if isequal(chd1,chd1_1)
    disp('      chd1��chd1_1��ͬ')
    disp(' ')
end   
if isequal(cvd1,cvd1_1)
    disp('      cvd1��cvd1_1��ͬ')
    disp(' ')
end  
if isequal(cdd1,cdd1_1)
    disp('      cdd1��cdd1_1��ͬ')
    disp(' ')
end   