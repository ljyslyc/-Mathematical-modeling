%% SOM����������ݷ���--���ͻ��������
% 
% 
% <html>
% <table border="0" width="600px" id="table1">	<tr>		<td><b><font size="2">�ð�������������</font></b></td>	</tr>	<tr><td><span class="comment"><font size="2">1�����˳���פ���ڴ�<a target="_blank" href="http://www.ilovematlab.cn/forum-158-1.html"><font color="#0000FF">���</font></a>��Ըð������ʣ��������ʱش𡣱����鼮�ٷ���վΪ��<a href="http://video.ourmatlab.com">video.ourmatlab.com</a></font></span></td></tr><tr>		<td><font size="2">2�����<a href="http://union.dangdang.com/transfer/transfer.aspx?from=P-284318&backurl=http://www.dangdang.com/">�ӵ���Ԥ������</a>��<a href="http://union.dangdang.com/transfer/transfer.aspx?from=P-284318&backurl=http://www.dangdang.com/">��Matlab������30������������</a>��</td></tr><tr>	<td><p class="comment"></font><font size="2">3</font><font size="2">���˰��������׵Ľ�ѧ��Ƶ����Ƶ���ط�ʽ<a href="http://video.ourmatlab.com/vbuy.html">video.ourmatlab.com/vbuy.html</a></font><font size="2">�� </font></p></td>	</tr>			<tr>		<td><span class="comment"><font size="2">		4���˰���Ϊԭ��������ת����ע����������Matlab������30����������������</font></span></td>	</tr>		<tr>		<td><span class="comment"><font size="2">		5�����˰��������������о��й��������ǻ�ӭ���������Ҫ��ȣ����ǿ��Ǻ���Լ��ڰ����</font></span></td>	</tr>		</table>
% </html>


%% ��ջ�������
clc
clear

%% ¼����������
% ��������
load p;

%ת�ú����������������ʽ
P=P';

%% ���罨����ѵ��
% newsom����SOM���硣minmax��P��ȡ����������Сֵ��������Ϊ6*6=36����Ԫ
net=newsom(minmax(P),[6 6]);
plotsom(net.layers{1}.positions)
% 5��ѵ���Ĳ���
a=[10 30 50 100 200 500 1000];
% �����ʼ��һ��1*10������
yc=rands(7,8);
%% ����ѵ��
% ѵ������Ϊ10��
net.trainparam.epochs=a(1);
% ѵ������Ͳ鿴����
net=train(net,P);
y=sim(net,P);
yc(1,:)=vec2ind(y);
plotsom(net.IW{1,1},net.layers{1}.distances)

% ѵ������Ϊ30��
net.trainparam.epochs=a(2);
% ѵ������Ͳ鿴����
net=train(net,P);
y=sim(net,P);
yc(2,:)=vec2ind(y);
plotsom(net.IW{1,1},net.layers{1}.distances)

% ѵ������Ϊ50��
net.trainparam.epochs=a(3);
% ѵ������Ͳ鿴����
net=train(net,P);
y=sim(net,P);
yc(3,:)=vec2ind(y);
plotsom(net.IW{1,1},net.layers{1}.distances)


% ѵ������Ϊ100��
net.trainparam.epochs=a(4);
% ѵ������Ͳ鿴����
net=train(net,P);
y=sim(net,P);
yc(4,:)=vec2ind(y);
plotsom(net.IW{1,1},net.layers{1}.distances)


% ѵ������Ϊ200��
net.trainparam.epochs=a(5);
% ѵ������Ͳ鿴����
net=train(net,P);
y=sim(net,P);
yc(5,:)=vec2ind(y);
plotsom(net.IW{1,1},net.layers{1}.distances)

% ѵ������Ϊ500��
net.trainparam.epochs=a(6);
% ѵ������Ͳ鿴����
net=train(net,P);
y=sim(net,P);
yc(6,:)=vec2ind(y);
plotsom(net.IW{1,1},net.layers{1}.distances)

% ѵ������Ϊ1000��
net.trainparam.epochs=a(7);
% ѵ������Ͳ鿴����
net=train(net,P);
y=sim(net,P);
yc(7,:)=vec2ind(y);
plotsom(net.IW{1,1},net.layers{1}.distances)
yc
%% �����������Ԥ��
% ������������
t=[0.9512 1.0000 0.9458 -0.4215 0.4218 0.9511 0.9645 0.8941]';
% sim( )�����������
r=sim(net,t);
% �任���� ����ֵ����ת����±�������
rr=vec2ind(r)

%% ������Ԫ�ֲ����
% �鿴��������ѧ�ṹ
plotsomtop(net)
% �鿴�ٽ���Ԫֱ�ӵľ������
plotsomnd(net)
% �鿴ÿ����Ԫ�ķ������
plotsomhits(net,P)

web browser http://www.matlabsky.com/thread-11162-1-1.html

%%
% <html>
% <table width="656" align="left" >	<tr><td align="center"><p><font size="2"><a href="http://video.ourmatlab.com/">Matlab������30����������</a></font></p><p align="left"><font size="2">�����̳��</font></p><p align="left"><font size="2">��Matlab������30�������������ٷ���վ��<a href="http://video.ourmatlab.com">video.ourmatlab.com</a></font></p><p align="left"><font size="2">Matlab������̳��<a href="http://www.matlabsky.com">www.matlabsky.com</a></font></p><p align="left"><font size="2">M</font><font size="2">atlab�����ٿƣ�<a href="http://www.mfun.la">www.mfun.la</a></font></p><p align="left"><font size="2">Matlab������̳��<a href="http://www.ilovematlab.com">www.ilovematlab.com</a></font></p></td>	</tr></table>
% </html>

