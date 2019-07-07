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
T_train = ind2vec(Tc_train);
% ���Լ�
test_label = rand_label(31:end);
P_test = pixel_value(test_label,:)';
Tc_test = direction_label(test_label);

%% ����LVQ����
for i = 1:5
    rate{i} = length(find(Tc_train == i))/30;
end
net = newlvq(minmax(P_train),20,cell2mat(rate),0.01,'learnlv1');
% ����ѵ������
net.trainParam.epochs = 100;
net.trainParam.goal = 0.001;
net.trainParam.lr = 0.1;

%% ѵ������
net = train(net,P_train,T_train);

%% ����ʶ�����
T_sim = sim(net,P_test);
Tc_sim = vec2ind(T_sim);
result = [Tc_test;Tc_sim]

%% �����ʾ
% ѵ�����������
strain_label = sort(train_label);
htrain_label = ceil(strain_label/N);
% ѵ��������������
dtrain_label = strain_label - floor(strain_label/N)*N;
dtrain_label(dtrain_label == 0) = N;
% ��ʾѵ����ͼ�����
disp('ѵ����ͼ��Ϊ��' );
for i = 1:30 
    str_train = [num2str(htrain_label(i)) '_'...
               num2str(dtrain_label(i)) '  '];
    fprintf('%s',str_train)
    if mod(i,5) == 0
        fprintf('\n');
    end
end
% ���Լ��������
stest_label = sort(test_label);
htest_label = ceil(stest_label/N);
% ���Լ�����������
dtest_label = stest_label - floor(stest_label/N)*N;
dtest_label(dtest_label == 0) = N;
% ��ʾ���Լ�ͼ�����
disp('���Լ�ͼ��Ϊ��');
for i = 1:20 
    str_test = [num2str(htest_label(i)) '_'...
              num2str(dtest_label(i)) '  '];
    fprintf('%s',str_test)
    if mod(i,5) == 0
        fprintf('\n');
    end
end
% ��ʾʶ�����ͼ��
error = Tc_sim - Tc_test;
location = {'��' '��ǰ��' 'ǰ��' '��ǰ��' '�ҷ�'};
for i = 1:length(error)
    if error(i) ~= 0
        % ʶ�����ͼ���������
        herror_label = ceil(test_label(i)/N);
        % ʶ�����ͼ������������
        derror_label = test_label(i) - floor(test_label(i)/N)*N;
        derror_label(derror_label == 0) = N;
        % ͼ��ԭʼ����
        standard = location{Tc_test(i)};
        % ͼ��ʶ��������
        identify = location{Tc_sim(i)};
        str_err = strcat(['ͼ��' num2str(herror_label) '_'...
                        num2str(derror_label) 'ʶ�����.']);
        disp([str_err '(��ȷ���������' standard...
                      '��ʶ����������' identify ')']);
    end
end
% ��ʾʶ����
disp(['ʶ����Ϊ��' num2str(length(find(error == 0))/20*100) '%']);

web browser http://www.matlabsky.com/thread-11193-1-1.html

%%
% <html>
% <table width="656" align="left" >	<tr><td align="center"><p><font size="2"><a href="http://video.ourmatlab.com/">Matlab������30����������</a></font></p><p align="left"><font size="2">�����̳��</font></p><p align="left"><font size="2">��Matlab������30�������������ٷ���վ��<a href="http://video.ourmatlab.com">video.ourmatlab.com</a></font></p><p align="left"><font size="2">Matlab������̳��<a href="http://www.matlabsky.com">www.matlabsky.com</a></font></p><p align="left"><font size="2">M</font><font size="2">atlab�����ٿƣ�<a href="http://www.mfun.la">www.mfun.la</a></font></p><p align="left"><font size="2">Matlab������̳��<a href="http://www.ilovematlab.com">www.ilovematlab.com</a></font></p></td>	</tr></table>
% </html>



        
        
        
