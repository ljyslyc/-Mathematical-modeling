%% ����7��RBF����Ļع�-�����Ժ����ع��ʵ�� 
% 
% 
% <html>
% <table border="0" width="600px" id="table1">	<tr>		<td><b><font size="2">�ð�������������</font></b></td>	</tr>	<tr><td><span class="comment"><font size="2">1�����˳���פ���ڴ�<a target="_blank" href="http://www.ilovematlab.cn/forum-158-1.html"><font color="#0000FF">���</font></a>��Ըð������ʣ��������ʱش𡣱����鼮�ٷ���վΪ��<a href="http://video.ourmatlab.com">video.ourmatlab.com</a></font></span></td></tr><tr>		<td><font size="2">2�����<a href="http://union.dangdang.com/transfer/transfer.aspx?from=P-284318&backurl=http://www.dangdang.com/">�ӵ���Ԥ������</a>��<a href="http://union.dangdang.com/transfer/transfer.aspx?from=P-284318&backurl=http://www.dangdang.com/">��Matlab������30������������</a>��</td></tr><tr>	<td><p class="comment"></font><font size="2">3</font><font size="2">���˰��������׵Ľ�ѧ��Ƶ����Ƶ���ط�ʽ<a href="http://video.ourmatlab.com/vbuy.html">video.ourmatlab.com/vbuy.html</a></font><font size="2">�� </font></p></td>	</tr>			<tr>		<td><span class="comment"><font size="2">		4���˰���Ϊԭ��������ת����ע����������Matlab������30����������������</font></span></td>	</tr>		<tr>		<td><span class="comment"><font size="2">		5�����˰��������������о��й��������ǻ�ӭ���������Ҫ��ȣ����ǿ��Ǻ���Լ��ڰ����</font></span></td>	</tr>		</table>
% </html>


%% ��ջ�������
clc
clear
%% ����ѵ��������ѵ�����룬ѵ�������
% ldΪ��������
ld=400; 

% ����2*ld�ľ��� 
x=rand(2,ld); 

% ��xת����[-1.5 1.5]֮��
x=(x-0.5)*1.5*2; 

% x�ĵ�һ��Ϊx1���ڶ���Ϊx2.
x1=x(1,:);
x2=x(2,:);

% �����������Fֵ
F=20+x1.^2-10*cos(2*pi*x1)+x2.^2-10*cos(2*pi*x2);

%% ����RBF������ 
% ����approximate RBF�����硣spreadΪĬ��ֵ
net=newrb(x,F);

%% ������������

% generate the testing data
interval=0.1;
[i, j]=meshgrid(-1.5:interval:1.5);
row=size(i);
tx1=i(:);
tx1=tx1';
tx2=j(:);
tx2=tx2';
tx=[tx1;tx2];

%% ʹ�ý�����RBF�������ģ�⣬�ó��������
ty=sim(net,tx);

%% ʹ��ͼ�񣬻���3άͼ

% �����ĺ���ͼ��
interval=0.1;
[x1, x2]=meshgrid(-1.5:interval:1.5);
F = 20+x1.^2-10*cos(2*pi*x1)+x2.^2-10*cos(2*pi*x2);
subplot(1,3,1)
mesh(x1,x2,F);
zlim([0,60])
title('�����ĺ���ͼ��')

% ����ó��ĺ���ͼ��
v=reshape(ty,row);
subplot(1,3,2)
mesh(i,j,v);
zlim([0,60])
title('RBF��������')


% ���ͼ��
subplot(1,3,3)
mesh(x1,x2,F-v);
zlim([0,60])
title('���ͼ��')

set(gcf,'position',[300 ,250,900,400])

web browser http://www.matlabsky.com/thread-11143-1-2.html
%%
% <html>
% <table width="656" align="left" >	<tr><td align="center"><p><font size="2"><a href="http://video.ourmatlab.com/">Matlab������30����������</a></font></p><p align="left"><font size="2">�����̳��</font></p><p align="left"><font size="2">��Matlab������30�������������ٷ���վ��<a href="http://video.ourmatlab.com">video.ourmatlab.com</a></font></p><p align="left"><font size="2">Matlab������̳��<a href="http://www.matlabsky.com">www.matlabsky.com</a></font></p><p align="left"><font size="2">M</font><font size="2">atlab�����ٿƣ�<a href="http://www.mfun.la">www.mfun.la</a></font></p><p align="left"><font size="2">Matlab������̳��<a href="http://www.ilovematlab.com">www.ilovematlab.com</a></font></p></td>	</tr></table>
% </html>


