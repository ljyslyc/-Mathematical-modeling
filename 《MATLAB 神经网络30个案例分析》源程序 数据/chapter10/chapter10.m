%% ��ɢHopfield�ķ��ࡪ����У������������
%
% <html>
% <table border="0" width="600px" id="table1">	<tr>		<td><b><font size="2">�ð�������������</font></b></td>	</tr>	<tr><td><span class="comment"><font size="2">1�����˳���פ���ڴ�<a target="_blank" href="http://www.ilovematlab.cn/forum-158-1.html"><font color="#0000FF">���</font></a>��Ըð������ʣ��������ʱش𡣱����鼮�ٷ���վΪ��<a href="http://video.ourmatlab.com">video.ourmatlab.com</a></font></span></td></tr><tr>		<td><font size="2">2�����<a href="http://union.dangdang.com/transfer/transfer.aspx?from=P-284318&backurl=http://www.dangdang.com/">�ӵ���Ԥ������</a>��<a href="http://union.dangdang.com/transfer/transfer.aspx?from=P-284318&backurl=http://www.dangdang.com/">��Matlab������30������������</a>��</td></tr><tr>	<td><p class="comment"></font><font size="2">3</font><font size="2">���˰��������׵Ľ�ѧ��Ƶ����Ƶ���ط�ʽ<a href="http://video.ourmatlab.com/vbuy.html">video.ourmatlab.com/vbuy.html</a></font><font size="2">�� </font></p></td>	</tr>			<tr>		<td><span class="comment"><font size="2">		4���˰���Ϊԭ��������ת����ע����������Matlab������30����������������</font></span></td>	</tr>		<tr>		<td><span class="comment"><font size="2">		5�����˰��������������о��й��������ǻ�ӭ���������Ҫ��ȣ����ǿ��Ǻ���Լ��ڰ����</font></span></td>	</tr>		</table>
% </html>
%

%% ��ջ�������
clear all
clc

%% ��������
load class.mat

%% Ŀ������
T = [class_1 class_2 class_3 class_4 class_5];

%% ��������
net = newhop(T);

%% �������������
load sim.mat
A = {[sim_1 sim_2 sim_3 sim_4 sim_5]};

%% �������
Y = sim(net,{25 20},{},A);

%% �����ʾ
Y1 = Y{20}(:,1:5)
Y2 = Y{20}(:,6:10)
Y3 = Y{20}(:,11:15)
Y4 = Y{20}(:,16:20)
Y5 = Y{20}(:,21:25)

%% ��ͼ
result = {T;A{1};Y{20}};
figure
for p = 1:3
    for k = 1:5 
        subplot(3,5,(p-1)*5+k)
        temp = result{p}(:,(k-1)*5+1:k*5);
        [m,n] = size(temp);
        for i = 1:m
            for j = 1:n
                if temp(i,j) > 0
                   plot(j,m-i,'ko','MarkerFaceColor','k');
                else
                   plot(j,m-i,'ko');
                end
                hold on
            end
        end
        axis([0 6 0 12])
        axis off
        if p == 1
           title(['class' num2str(k)])
        elseif p == 2
           title(['pre-sim' num2str(k)])
        else
           title(['sim' num2str(k)])
        end
    end                
end

% ������չ(�޷��ֱ����)
noisy = [1 -1 -1 -1 -1;-1 -1 -1 1 -1;
        -1 1 -1 -1 -1;-1 1 -1 -1 -1;
        1 -1 -1 -1 -1;-1 -1 1 -1 -1;
        -1 -1 -1 1 -1;-1 -1 -1 -1 1;
        -1 1 -1 -1 -1;-1 -1 -1 1 -1;
        -1 -1 1 -1 -1];
y = sim(net,{5 100},{},{noisy});
a = y{100}

web browser http://www.matlabsky.com/thread-11146-1-2.html
%%
% <html>
% <table width="656" align="left" >	<tr><td align="center"><p><font size="2"><a href="http://video.ourmatlab.com/">Matlab������30����������</a></font></p><p align="left"><font size="2">�����̳��</font></p><p align="left"><font size="2">��Matlab������30�������������ٷ���վ��<a href="http://video.ourmatlab.com">video.ourmatlab.com</a></font></p><p align="left"><font size="2">Matlab������̳��<a href="http://www.matlabsky.com">www.matlabsky.com</a></font></p><p align="left"><font size="2">M</font><font size="2">atlab�����ٿƣ�<a href="http://www.mfun.la">www.mfun.la</a></font></p><p align="left"><font size="2">Matlab������̳��<a href="http://www.ilovematlab.com">www.ilovematlab.com</a></font></p></td>	</tr></table>
% </html>




