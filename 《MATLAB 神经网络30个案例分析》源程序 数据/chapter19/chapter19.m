%%  ����19: ����������ķ���Ԥ��--����PNN�ı�ѹ���������
% 
%
% <html>
% <table border="0" width="600px" id="table1">	<tr>		<td><b><font size="2">�ð�������������</font></b></td>	</tr>	<tr><td><span class="comment"><font size="2">1�����˳���פ���ڴ�<a target="_blank" href="http://www.ilovematlab.cn/forum-158-1.html"><font color="#0000FF">���</font></a>��Ըð������ʣ��������ʱش𡣱����鼮�ٷ���վΪ��<a href="http://video.ourmatlab.com">video.ourmatlab.com</a></font></span></td></tr><tr>		<td><font size="2">2�����<a href="http://union.dangdang.com/transfer/transfer.aspx?from=P-284318&backurl=http://www.dangdang.com/">�ӵ���Ԥ������</a>��<a href="http://union.dangdang.com/transfer/transfer.aspx?from=P-284318&backurl=http://www.dangdang.com/">��Matlab������30������������</a>��</td></tr><tr>	<td><p class="comment"></font><font size="2">3</font><font size="2">���˰��������׵Ľ�ѧ��Ƶ����Ƶ���ط�ʽ<a href="http://video.ourmatlab.com/vbuy.html">video.ourmatlab.com/vbuy.html</a></font><font size="2">�� </font></p></td>	</tr>			<tr>		<td><span class="comment"><font size="2">		4���˰���Ϊԭ��������ת����ע����������Matlab������30����������������</font></span></td>	</tr>		<tr>		<td><span class="comment"><font size="2">		5�����˰��������������о��й��������ǻ�ӭ���������Ҫ��ȣ����ǿ��Ǻ���Լ��ڰ����</font></span></td>	</tr>		</table>
% </html>




%% ��ջ�������
clc;
clear all
close all
nntwarn off;
warning off;
%% ��������
load data
%% ѡȡѵ�����ݺͲ�������

Train=data(1:23,:);
Test=data(24:end,:);
p_train=Train(:,1:3)';
t_train=Train(:,4)';
p_test=Test(:,1:3)';
t_test=Test(:,4)';

%% ���������ת��Ϊ����
t_train=ind2vec(t_train);
t_train_temp=Train(:,4)';
%% ʹ��newpnn��������PNN SPREADѡȡΪ1.5
Spread=1.5;
net=newpnn(p_train,t_train,Spread)

%% ѵ�����ݻش� �鿴����ķ���Ч��

% Sim������������Ԥ��
Y=sim(net,p_train);
% �������������ת��Ϊָ��
Yc=vec2ind(Y);

%% ͨ����ͼ �۲������ѵ�����ݷ���Ч��
figure(1)
subplot(1,2,1)
stem(1:length(Yc),Yc,'bo')
hold on
stem(1:length(Yc),t_train_temp,'r*')
title('PNN ����ѵ�����Ч��')
xlabel('�������')
ylabel('������')
set(gca,'Ytick',[1:5])
subplot(1,2,2)
H=Yc-t_train_temp;
stem(H)
title('PNN ����ѵ��������ͼ')
xlabel('�������')


%% ����Ԥ��δ֪����Ч��
Y2=sim(net,p_test);
Y2c=vec2ind(Y2);
figure(2)
stem(1:length(Y2c),Y2c,'b^')
hold on
stem(1:length(Y2c),t_test,'r*')
title('PNN �����Ԥ��Ч��')
xlabel('Ԥ���������')
ylabel('������')
set(gca,'Ytick',[1:5])
web browser http://www.matlabsky.com/thread-11164-1-1.html

%%
% <html>
% <table width="656" align="left" >	<tr><td align="center"><p><font size="2"><a href="http://video.ourmatlab.com/">Matlab������30����������</a></font></p><p align="left"><font size="2">�����̳��</font></p><p align="left"><font size="2">��Matlab������30�������������ٷ���վ��<a href="http://video.ourmatlab.com">video.ourmatlab.com</a></font></p><p align="left"><font size="2">Matlab������̳��<a href="http://www.matlabsky.com">www.matlabsky.com</a></font></p><p align="left"><font size="2">M</font><font size="2">atlab�����ٿƣ�<a href="http://www.mfun.la">www.mfun.la</a></font></p><p align="left"><font size="2">Matlab������̳��<a href="http://www.ilovematlab.com">www.ilovematlab.com</a></font></p></td>	</tr></table>
% </html>


