%% ϵͳ���෨��һ������
clc, clear

a = [1 0;
     1 1;
     3 2;
     4 3;
     2 5];
[m, n] = size(a);
y = pdist(a, 'cityblock');  % ��a����m����СΪn�����������ɰ���������Ϣ������
yc = squareform(y)          % ����������ת��Ϊ����
z = linkage(y)              % ʹ����̾��뷨���ɾ�����
[h, t] = dendrogram(z)