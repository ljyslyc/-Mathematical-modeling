clc,clear
ob=VideoReader('test.avi')     %��ȡ��Ƶ�ļ�����
get(ob) %��ȡ��Ƶ����Ĳ���
n=ob.NumberOfFrame;  %��ȡ��Ƶ����֡��
for i=1:n
    a=read(ob,i); %��ȡ��Ƶ����ĵ�i֡
    imshow(a)  %��ʾ��i֡ͼ��
    str=['source\',int2str(i),'.jpg']; %�����ļ������ַ�����Ŀ¼sourceҪ��ǰ����
    imwrite(a,str); %�ѵ�i֡���浽jpg�ļ�
end
