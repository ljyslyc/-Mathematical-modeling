close all;                  %�رյ�ǰ����ͼ�δ��ڣ���չ����ռ��������������ռ����б���
clear all;
clc;
load woman;         %��ȡͼ������
nbcol=size(map,1);
[c,s]=wavedec2(X,2,'db2');%����db4С������2��ͼ��ֽ�
siz=s(size(s,1),:);       %��ȡԭͼ�����X�Ĵ�С
ca2=appcoef2(c,s,'db2',2);%��ȡ���С���ֽ�ṹC��S�ĵ�1��С���任�Ľ���ϵ��
chd2=detcoef2('h',c,s,2); %���õĶ��С���ֽ�ṹC��S����ȡͼ���1���ϸ��ϵ����ˮƽ����
cvd2=detcoef2('v',c,s,2); %���õĶ��С���ֽ�ṹC��S����ȡͼ���1���ϸ��ϵ���Ĵ�ֱ����
cdd2=detcoef2('d',c,s,2); %���õĶ��С���ֽ�ṹC��S����ȡͼ���1���ϸ��ϵ���ĶԽǷ���
chd1=detcoef2('h',c,s,1); %���õĶ��С���ֽ�ṹC��S����ȡͼ���1���ϸ��ϵ����ˮƽ����
cvd1=detcoef2('v',c,s,1); %���õĶ��С���ֽ�ṹC��S����ȡͼ���1���ϸ��ϵ���Ĵ�ֱ����
cdd1=detcoef2('d',c,s,1); %���õĶ��С���ֽ�ṹC��S����ȡͼ���1���ϸ��ϵ���ĶԽǷ���
ca11=ca2+chd2+cvd2+cdd2;  %�����ع�����ͼ��          
ca1 = appcoef2(c,s,'db4',1);%��ȡ���С���ֽ�ṹC��S�ĵ�1��С���任�Ľ���ϵ��
set(0,'defaultFigurePosition',[100,100,1000,500]);%�޸�ͼ��ͼ��λ�õ�Ĭ������
set(0,'defaultFigureColor',[1 1 1])%�޸�ͼ�α�����ɫ������
figure                                             %��ʾͼ����
subplot(1,4,1); imshow(uint8(wcodemat(ca2,nbcol)));
subplot(1,4,2); imshow(uint8(wcodemat(chd2,nbcol)));
subplot(1,4,3); imshow(uint8(wcodemat(cvd2,nbcol)));
subplot(1,4,4); imshow(uint8(wcodemat(cdd2,nbcol)));
figure
subplot(1,4,1); imshow(uint8(wcodemat(ca11,nbcol)));
subplot(1,4,2); imshow(uint8(wcodemat(chd1,nbcol)));
subplot(1,4,3); imshow(uint8(wcodemat(cvd1,nbcol)));
subplot(1,4,4); imshow(uint8(wcodemat(cdd1,nbcol)));
disp('С������ֽ�Ľ���ϵ������ca2�Ĵ�С��')  %��ʾС���ֽ�ϵ������Ĵ�С
ca2_size=s(1,:)
disp('С������ֽ��ϸ��ϵ������cd2�Ĵ�С:')
cd2_size=s(2,:)
disp('С��һ��ֽ��ϸ��ϵ������cd1�Ĵ�С:')
cd1_size=s(3,:)
disp('ԭͼ���С:')
X_size=s(4,:)
disp('С���ֽ�ϵ����������c�ĳ���:')
c_size=length(c)