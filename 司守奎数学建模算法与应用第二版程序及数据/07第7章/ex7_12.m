clc, clear
x=[41.5	42.0	40.0	42.5	42.0	42.2	42.7	42.1	41.4];
y=[41.2	41.8	42.4	41.6	41.7	41.3];
yx=[y,x]; yxr=tiedrank(yx) %������
yr=sum(yxr(1:length(y))) %����y���Ⱥ�
[p,h,s]=ranksum(y,x) %����Matlab������ֱ�ӽ��м���
