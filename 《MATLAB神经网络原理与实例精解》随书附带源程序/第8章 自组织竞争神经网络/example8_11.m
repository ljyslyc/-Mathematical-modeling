% example8_11.m
x = simplecluster_dataset;
plot(x(1,:),x(2,:),'o')
set(gcf,'color','w')
title('ԭʼ����')


net = selforgmap([8 8]);		% ��������֯ӳ������
net = train(net,x);				% ѵ��
y = net(x);
classes = vec2ind(y);
hist(classes,64)				% ��ʾ������
set(gcf,'color','w')
title('������')
xlabel('���')
ylabel('����������������')

net = selforgmap([2,3]);
net = train(net,x);
y = net(x);
classes = vec2ind(y);
c=hist(classes,6)			% 6������������������
plotsomhits(net,x)          % ��ʾÿ�����ĸ���
plotsompos(net,x)           % ��ʾ������ĵ��λ��


web -broswer http://www.ilovematlab.cn/forum-222-1.html