% example8_14.m

x=[1,2,3;1,2,4]					% ����һ������
[xx,settings]=mapminmax(x);		% ��һ����[0,1]
xx
[settings.xmin,settings.xmax]		% �ṹ��settings�б�����ÿ�е������Сֵ
fp.ymin=0;fp.ymax=10	
[xx,settings]=mapminmax(x,fp);		% ӳ�䵽[0,10]����
xx
[xx,settings]=mapminmax(x',fp);		% ���н��й�һ��
xx
web -broswer http://www.ilovematlab.cn/forum-222-1.html