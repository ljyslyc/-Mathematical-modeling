%% ����Hopfield��������Ż��������������Ż�����
%
% <html>
% <table border="0" width="600px" id="table1">	<tr>		<td><b><font size="2">�ð�������������</font></b></td>	</tr>	<tr><td><span class="comment"><font size="2">1�����˳���פ���ڴ�<a target="_blank" href="http://www.ilovematlab.cn/forum-158-1.html"><font color="#0000FF">���</font></a>��Ըð������ʣ��������ʱش𡣱����鼮�ٷ���վΪ��<a href="http://video.ourmatlab.com">video.ourmatlab.com</a></font></span></td></tr><tr>		<td><font size="2">2�����<a href="http://union.dangdang.com/transfer/transfer.aspx?from=P-284318&backurl=http://www.dangdang.com/">�ӵ���Ԥ������</a>��<a href="http://union.dangdang.com/transfer/transfer.aspx?from=P-284318&backurl=http://www.dangdang.com/">��Matlab������30������������</a>��</td></tr><tr>	<td><p class="comment"></font><font size="2">3</font><font size="2">���˰��������׵Ľ�ѧ��Ƶ����Ƶ���ط�ʽ<a href="http://video.ourmatlab.com/vbuy.html">video.ourmatlab.com/vbuy.html</a></font><font size="2">�� </font></p></td>	</tr>			<tr>		<td><span class="comment"><font size="2">		4���˰���Ϊԭ��������ת����ע����������Matlab������30����������������</font></span></td>	</tr>		<tr>		<td><span class="comment"><font size="2">		5�����˰��������������о��й��������ǻ�ӭ���������Ҫ��ȣ����ǿ��Ǻ���Լ��ڰ����</font></span></td>	</tr>		</table>
% </html>

%% ��ջ�������������ȫ�ֱ���
clear all
clc
global A D

%% �������λ��
load city_location

%% �����໥���м����
distance = dist(citys,citys');

%% ��ʼ������
N = size(citys,1);
A = 200;
D = 100;
U0 = 0.1;
step = 0.0001;
delta = 2 * rand(N,N) - 1;
U = U0 * log(N-1) + delta;
V = (1 + tansig(U/U0))/2;
iter_num = 10000;
E = zeros(1,iter_num);

%% Ѱ�ŵ���
for k = 1:iter_num  
    % ��̬���̼���
    dU = diff_u(V,distance);
    % ������Ԫ״̬����
    U = U + dU*step;
    % �����Ԫ״̬����
    V = (1 + tansig(U/U0))/2;
    % ������������
    e = energy(V,distance);
    E(k) = e;  
end

 %% �ж�·����Ч��
[rows,cols] = size(V);
V1 = zeros(rows,cols);
[V_max,V_ind] = max(V);
for j = 1:cols
    V1(V_ind(j),j) = 1;
end
C = sum(V1,1);
R = sum(V1,2);
flag = isequal(C,ones(1,N)) & isequal(R',ones(1,N));

%% �����ʾ
if flag == 1
   % �����ʼ·������
   sort_rand = randperm(N);
   citys_rand = citys(sort_rand,:);
   Length_init = dist(citys_rand(1,:),citys_rand(end,:)');
   for i = 2:size(citys_rand,1)
       Length_init = Length_init+dist(citys_rand(i-1,:),citys_rand(i,:)');
   end
   % ���Ƴ�ʼ·��
   figure(1)
   plot([citys_rand(:,1);citys_rand(1,1)],[citys_rand(:,2);citys_rand(1,2)],'o-')
   for i = 1:length(citys)
       text(citys(i,1),citys(i,2),['   ' num2str(i)])
   end
   text(citys_rand(1,1),citys_rand(1,2),['       ���' ])
   text(citys_rand(end,1),citys_rand(end,2),['       �յ�' ])
   title(['�Ż�ǰ·��(���ȣ�' num2str(Length_init) ')'])
   axis([0 1 0 1])
   grid on
   xlabel('����λ�ú�����')
   ylabel('����λ��������')
   % ��������·������
   [V1_max,V1_ind] = max(V1);
   citys_end = citys(V1_ind,:);
   Length_end = dist(citys_end(1,:),citys_end(end,:)');
   for i = 2:size(citys_end,1)
       Length_end = Length_end+dist(citys_end(i-1,:),citys_end(i,:)');
   end
   disp('����·������');V1
   % ��������·��
   figure(2)
   plot([citys_end(:,1);citys_end(1,1)],...
       [citys_end(:,2);citys_end(1,2)],'o-')
   for i = 1:length(citys)
       text(citys(i,1),citys(i,2),['  ' num2str(i)])
   end
   text(citys_end(1,1),citys_end(1,2),['       ���' ])
   text(citys_end(end,1),citys_end(end,2),['       �յ�' ])
   title(['�Ż���·��(���ȣ�' num2str(Length_end) ')'])
   axis([0 1 0 1])
   grid on
   xlabel('����λ�ú�����')
   ylabel('����λ��������')
   % �������������仯����
   figure(3)
   plot(1:iter_num,E);
   ylim([0 2000])
   title(['���������仯����(����������' num2str(E(end)) ')']);
   xlabel('��������');
   ylabel('��������');
else
   disp('Ѱ��·����Ч');
end

web browser http://www.matlabsky.com/thread-11156-1-2.html

%%
% <html>
% <table width="656" align="left" >	<tr><td align="center"><p><font size="2"><a href="http://video.ourmatlab.com/">Matlab������30����������</a></font></p><p align="left"><font size="2">�����̳��</font></p><p align="left"><font size="2">��Matlab������30�������������ٷ���վ��<a href="http://video.ourmatlab.com">video.ourmatlab.com</a></font></p><p align="left"><font size="2">Matlab������̳��<a href="http://www.matlabsky.com">www.matlabsky.com</a></font></p><p align="left"><font size="2">M</font><font size="2">atlab�����ٿƣ�<a href="http://www.mfun.la">www.mfun.la</a></font></p><p align="left"><font size="2">Matlab������̳��<a href="http://www.ilovematlab.com">www.ilovematlab.com</a></font></p></td>	</tr></table>
% </html>
