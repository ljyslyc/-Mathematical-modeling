%% LVQ�������Ԥ�⡪������ʶ��
%
% <html>
% <table border="0" width="600px" id="table1">	<tr>		<td><b><font size="2">�ð�������������</font></b></td>	</tr>	<tr><td><span class="comment"><font size="2">1�����˳���פ���ڴ�<a target="_blank" href="http://www.ilovematlab.cn/forum-158-1.html"><font color="#0000FF">���</font></a>��Ըð������ʣ��������ʱش𡣱����鼮�ٷ���վΪ��<a href="http://video.ourmatlab.com">video.ourmatlab.com</a></font></span></td></tr><tr>		<td><font size="2">2�����<a href="http://union.dangdang.com/transfer/transfer.aspx?from=P-284318&backurl=http://www.dangdang.com/">�ӵ���Ԥ������</a>��<a href="http://union.dangdang.com/transfer/transfer.aspx?from=P-284318&backurl=http://www.dangdang.com/">��Matlab������30������������</a>��</td></tr><tr>	<td><p class="comment"></font><font size="2">3</font><font size="2">���˰��������׵Ľ�ѧ��Ƶ����Ƶ���ط�ʽ<a href="http://video.ourmatlab.com/vbuy.html">video.ourmatlab.com/vbuy.html</a></font><font size="2">�� </font></p></td>	</tr>			<tr>		<td><span class="comment"><font size="2">		4���˰���Ϊԭ��������ת����ע����������Matlab������30����������������</font></span></td>	</tr>		<tr>		<td><span class="comment"><font size="2">		5�����˰��������������о��й��������ǻ�ӭ���������Ҫ��ȣ����ǿ��Ǻ���Լ��ڰ����</font></span></td>	</tr>		</table>
% </html>

%% �����������
clear all
clc

%% ��������������ȡ 
% ����
M = 10;
% �������������
N = 5; 
% ����������ȡ
pixel_value = feature_extraction(M,N);

%% ѵ����/���Լ�����
% ����ͼ����ŵ��������
rand_label = randperm(M*N);  
% ����������
direction_label = repmat(1:N,1,M);
% ѵ����
train_label = rand_label(1:30);
P_train = pixel_value(train_label,:)';
Tc_train = direction_label(train_label);
% ���Լ�
test_label = rand_label(31:end);
P_test = pixel_value(test_label,:)';
Tc_test = direction_label(test_label);

%% ����PC
for i = 1:5
    rate{i} = length(find(Tc_train == i))/30;
end

%% LVQ1�㷨
[w1,w2] = lvq1_train(P_train,Tc_train,20,cell2mat(rate),0.01,5);
result_1 = lvq_predict(P_test,Tc_test,20,w1,w2);

%% LVQ2�㷨
[w1,w2] = lvq2_train(P_train,Tc_train,20,0.01,5,w1,w2);
result_2 = lvq_predict(P_test,Tc_test,20,w1,w2);

web browser http://www.matlabsky.com/thread-11193-1-1.html

%%
% <html>
% <table width="656" align="left" >	<tr><td align="center"><p><font size="2"><a href="http://video.ourmatlab.com/">Matlab������30����������</a></font></p><p align="left"><font size="2">�����̳��</font></p><p align="left"><font size="2">��Matlab������30�������������ٷ���վ��<a href="http://video.ourmatlab.com">video.ourmatlab.com</a></font></p><p align="left"><font size="2">Matlab������̳��<a href="http://www.matlabsky.com">www.matlabsky.com</a></font></p><p align="left"><font size="2">M</font><font size="2">atlab�����ٿƣ�<a href="http://www.mfun.la">www.mfun.la</a></font></p><p align="left"><font size="2">Matlab������̳��<a href="http://www.ilovematlab.com">www.ilovematlab.com</a></font></p></td>	</tr></table>
% </html>