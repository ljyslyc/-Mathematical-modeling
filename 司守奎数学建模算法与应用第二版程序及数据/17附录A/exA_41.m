clc, clear
tf=dir('*.txt') %������ı��ļ�����Ϣ������ֵ�ǽṹ����
n=length(tf); %���㴿�ı��ļ��ĸ���
fts=ascii2fts(tf(1).name);  %����һ���ļ��е�ʱ����������
fts=extfield(fts,{'series2','series3'}); %�����2���ֶκ͵�3���ֶ�
for i=2:n
    tp1=ascii2fts(tf(i).name); %��ʱ����������
    tp2=extfield(tp1,{'series2','series3'}); %�����2,3�ֶ�
    str1=['series',num2str(2*i)]; str2=['series',num2str(2*i+1)];
    tp3=fints(tp2.dates,fts2mat(tp2),{str1,str2}); %��ʱ�����и���
    fts=merge(fts,tp3); %�ϲ�����ʱ�����е�����
end
fts %��ʾ�ϲ�����������ֶ�����
