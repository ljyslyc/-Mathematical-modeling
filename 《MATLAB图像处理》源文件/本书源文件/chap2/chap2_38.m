close all; clear all; clc;					%�ر�����ͼ�δ��ڣ���������ռ����б��������������
k=5;
for m = 1:k							%����һ��5�׷���A�����б���б���ȵ�Ԫ�ظ�2��
    for n = 1:k							%�б���б�Ĳ�ľ���ֵΪ2��Ԫ�ظ�1
        if m == n
            a(m,n) = 2;
        elseif abs(m-n) == 2
            a(m,n) = 1;
        else
            a(m,n) = 0;
        end
    end
end
