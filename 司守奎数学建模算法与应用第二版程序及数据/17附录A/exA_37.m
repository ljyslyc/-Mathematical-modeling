clc, clear
n=5  %�ļ�����
mydata=cell(1, n); %��ʼ����Ÿ����ļ����ݵ�ϸ������
for k=1:n
    filename=sprintf('book%d.xls', k); %�����ļ����ĸ�ʽ���ַ���
    mydata{k}=importdata(filename); %���ļ���������
end
celldisp(mydata) %��ʾϸ�����������