% compare_rbf_grnn.m
x=-9:8;                                         % ������xֵ
y=[129,-32,-118,-138,-125,-97,-55,-23,-4,...    % yֵ
2,1,-31,-72,-121,-142,-174,-155,-77];
plot(x,y,'o')
P=x;
T=y;
tic;net = newrb(P, T, 0, 2);toc                 % �������������

xx=-9:.2:8;
yy = sim(net, xx);                              % ������������
figure(1);
hold on;
plot(xx,yy)
tic;net2=newgrnn(P,T,.5);toc;                   % ��ƹ���ع�����

yy2 = sim(net, xx);                             % ����ع��������
plot(xx,yy2,'.-r');
