close all; clear all; clc;					%�ر�����ͼ�δ��ڣ���������ռ����б��������������
k = 5;								
hilbert = zeros(k,k);     					%����һ��5��5ȫ0����
for m = 1:k							%Ӧ��for��Hilbert����ֵ
    for n = 1:k
        hilbert(m,n) = 1/(m+n -1);
    end
end
format rat
