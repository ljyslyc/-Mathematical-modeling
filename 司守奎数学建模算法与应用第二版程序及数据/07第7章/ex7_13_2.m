clc, clear
a=[18.2  9.5  12.0  21.1  10.2]; %����ԭʼ����
b=bootstrp(1000,@(x)quantile(x,0.5),a) %�������bootstrap��������λ��
c=std(b)  %������λ����׼��
