clc, clear
fi=dir('*.xls') %���Excel�ļ�����Ϣ������ֵ�ǽṹ����
n=length(fi); %����Excel�ļ��ĸ���
myData=cell(1,n);
for k=1:n
    myData{k}=importdata(fi(k).name);
end
celldisp(myData) %��ʾϸ�����������
