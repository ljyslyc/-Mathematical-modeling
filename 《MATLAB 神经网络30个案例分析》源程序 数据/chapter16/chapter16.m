%% ����16�����㾺������������ݷ��ࡪ���߰�֢����Ԥ��
% 
% <html>
% <table border="0" width="600px" id="table1">	<tr>		<td><b><font size="2">�ð�������������</font></b></td>	</tr>	<tr><td><span class="comment"><font size="2">1�����˳���פ���ڴ�<a target="_blank" href="http://www.ilovematlab.cn/forum-158-1.html"><font color="#0000FF">���</font></a>��Ըð������ʣ��������ʱش𡣱����鼮�ٷ���վΪ��<a href="http://video.ourmatlab.com">video.ourmatlab.com</a></font></span></td></tr><tr>		<td><font size="2">2�����<a href="http://union.dangdang.com/transfer/transfer.aspx?from=P-284318&backurl=http://www.dangdang.com/">�ӵ���Ԥ������</a>��<a href="http://union.dangdang.com/transfer/transfer.aspx?from=P-284318&backurl=http://www.dangdang.com/">��Matlab������30������������</a>��</td></tr><tr>	<td><p class="comment"></font><font size="2">3</font><font size="2">���˰��������׵Ľ�ѧ��Ƶ����Ƶ���ط�ʽ<a href="http://video.ourmatlab.com/vbuy.html">video.ourmatlab.com/vbuy.html</a></font><font size="2">�� </font></p></td>	</tr>			<tr>		<td><span class="comment"><font size="2">		4���˰���Ϊԭ��������ת����ע����������Matlab������30����������������</font></span></td>	</tr>		<tr>		<td><span class="comment"><font size="2">		5�����˰��������������о��й��������ǻ�ӭ���������Ҫ��ȣ����ǿ��Ǻ���Լ��ڰ����</font></span></td>	</tr>		</table>
% </html>




%% ��ջ�������
clc
clear

%% ¼����������
% �������ݲ������ݷֳ�ѵ����Ԥ������
load gene.mat;
data=gene;
P=data(1:40,:);
T=data(41:60,:);

% ת�ú����������������ʽ
P=P';
T=T';
% ȡ����Ԫ�ص����ֵ����СֵQ��
Q=minmax(P);

%% ���罨����ѵ��
% ����newc( )������������磺2�����������Ԫ������Ҳ����Ҫ����ĸ�����0.1����ѧϰ���ʡ�
net=newc(Q,2,0.1)

% ��ʼ�����缰�趨���������
net=init(net);
net.trainparam.epochs=20;
% ѵ�����磺
net=train(net,P);


%% �����Ч����֤

% ��ԭ���ݻش�����������Ч����
a=sim(net,P);
ac=vec2ind(a)

% ����ʹ���˱任����vec2ind()�����ڽ���ֵ������任���±�����������õĸ�ʽΪ��
%  ind=vec2ind(vec)
% ���У�
% vec��Ϊm��n�е���������x��x�е�ÿ��������i��������һ��1�⣬����Ԫ�ؾ�Ϊ0��
% ind��Ϊn��Ԫ��ֵΪ1���ڵ����±�ֵ���ɵ�һ����������



%% �����������Ԥ��
% ���潫��20�����ݴ���������ģ���У��۲����������
% sim( )�����������
Y=sim(net,T)
yc=vec2ind(Y)

web browser http://www.matlabsky.com/thread-11161-1-2.html

%%
% <html>
% <table width="656" align="left" >	<tr><td align="center"><p><font size="2"><a href="http://video.ourmatlab.com/">Matlab������30����������</a></font></p><p align="left"><font size="2">�����̳��</font></p><p align="left"><font size="2">��Matlab������30�������������ٷ���վ��<a href="http://video.ourmatlab.com">video.ourmatlab.com</a></font></p><p align="left"><font size="2">Matlab������̳��<a href="http://www.matlabsky.com">www.matlabsky.com</a></font></p><p align="left"><font size="2">M</font><font size="2">atlab�����ٿƣ�<a href="http://www.mfun.la">www.mfun.la</a></font></p><p align="left"><font size="2">Matlab������̳��<a href="http://www.ilovematlab.com">www.ilovematlab.com</a></font></p></td>	</tr></table>
% </html>
