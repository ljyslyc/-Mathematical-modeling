%%  ����18: ����Elman������ĵ�������Ԥ��ģ���о�
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

%% ��������

load data;
a=data;

%% ѡȡѵ�����ݺͲ�������

for i=1:6
    p(i,:)=[a(i,:),a(i+1,:),a(i+2,:)];
end
% ѵ����������
p_train=p(1:5,:);
% ѵ���������
t_train=a(4:8,:);
% ������������
p_test=p(6,:);
% �����������
t_test=a(9,:);

% Ϊ��Ӧ����ṹ ��ת��

p_train=p_train';
t_train=t_train';
p_test=p_test';


%% ����Ľ�����ѵ��
% ����ѭ�������ò�ͬ�����ز���Ԫ����
nn=[7 11 14 18];
for i=1:4
    threshold=[0 1;0 1;0 1;0 1;0 1;0 1;0 1;0 1;0 1];
    % ����Elman������ ���ز�Ϊnn(i)����Ԫ
    net=newelm(threshold,[nn(i),3],{'tansig','purelin'});
    % ��������ѵ������
    net.trainparam.epochs=1000;
    net.trainparam.show=20;
    % ��ʼ������
    net=init(net);
    % Elman����ѵ��
    net=train(net,p_train,t_train);
    % Ԥ������
    y=sim(net,p_test);
    % �������
    error(i,:)=y'-t_test;
end

%% ͨ����ͼ �۲첻ͬ���ز���Ԫ����ʱ�������Ԥ��Ч��

plot(1:1:3,error(1,:),'-ro','linewidth',2);
hold on;
plot(1:1:3,error(2,:),'b:x','linewidth',2);
hold on;
plot(1:1:3,error(3,:),'k-.s','linewidth',2);
hold on;
plot(1:1:3,error(4,:),'c--d','linewidth',2);
title('ElmanԤ�����ͼ')
set(gca,'Xtick',[1:3])
legend('7','11','14','18','location','best')
xlabel('ʱ���')
ylabel('���')
hold off;

web browser http://www.matlabsky.com/thread-11163-1-1.html
%%
%
%%
% <html>
% <table width="656" align="left" >	<tr><td align="center"><p><font size="2"><a href="http://video.ourmatlab.com/">Matlab������30����������</a></font></p><p align="left"><font size="2">�����̳��</font></p><p align="left"><font size="2">��Matlab������30�������������ٷ���վ��<a href="http://video.ourmatlab.com">video.ourmatlab.com</a></font></p><p align="left"><font size="2">Matlab������̳��<a href="http://www.matlabsky.com">www.matlabsky.com</a></font></p><p align="left"><font size="2">M</font><font size="2">atlab�����ٿƣ�<a href="http://www.mfun.la">www.mfun.la</a></font></p><p align="left"><font size="2">Matlab������̳��<a href="http://www.ilovematlab.com">www.ilovematlab.com</a></font></p></td>	</tr></table>
% </html>
