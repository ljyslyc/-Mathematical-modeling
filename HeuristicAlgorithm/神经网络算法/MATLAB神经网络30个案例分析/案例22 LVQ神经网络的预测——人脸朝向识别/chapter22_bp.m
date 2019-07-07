%% BP神经网络的预测——人脸识别
%
% <html>
% <table border="0" width="600px" id="table1">	<tr>		<td><b><font size="2">该案例作者申明：</font></b></td>	</tr>	<tr>		<td><span class="comment"><font size="2">1：本人长期驻扎在此<a target="_blank" href="http://www.ilovematlab.cn/forum-158-1.html"><font color="#0000FF">板块</font></a>里，对<a target="_blank" href="http://www.ilovematlab.cn/thread-49221-1-1.html"><font color="#0000FF">该案例</font></a>提问，做到有问必答。</font></span></td></tr><tr>	<td><span class="comment"><font size="2">2：此案例有配套的教学视频，配套的完整可运行Matlab程序。</font></span></td>	</tr>	<tr>		<td><span class="comment"><font size="2">		3：以下内容为该案例的部分内容（约占该案例完整内容的1/10）。</font></span></td>	</tr>		<tr>		<td><span class="comment"><font size="2">		4：此案例为原创案例，转载请注明出处（<a target="_blank" href="http://www.ilovematlab.cn/">Matlab中文论坛</a>，<a target="_blank" href="http://www.ilovematlab.cn/forum-158-1.html">《Matlab神经网络30个案例分析》</a>）。</font></span></td>	</tr>		<tr>		<td><span class="comment"><font size="2">		5：若此案例碰巧与您的研究有关联，我们欢迎您提意见，要求等，我们考虑后可以加在案例里。</font></span></td>	</tr>		<tr>		<td><span class="comment"><font size="2">		6：您看到的以下内容为初稿，书籍的实际内容可能有少许出入，以书籍实际发行内容为准。</font></span></td>	</tr><tr>		<td><span class="comment"><font size="2">		7：此书其他常见问题、预定方式等，<a target="_blank" href="http://www.ilovematlab.cn/thread-47939-1-1.html">请点击这里</a>。</font></span></td>	</tr></table>
% </html>
%
web browser http://www.ilovematlab.cn/thread-61927-1-1.html
%% 清除环境变量
clear all
clc;
%% 人脸特征向量提取 
% 人数
M=10;
% 人脸朝向类别数
N=5; 
% 特征向量提取
pixel_value=feature_extraction(M,N);
%% 训练集/测试集产生
% 产生图像序号的随机序列
rand_label=randperm(M*N);  
% 人脸朝向标号
direction_label=[1 0 0;1 1 0;0 1 0;0 1 1;0 0 1];
% 训练集
train_label=rand_label(1:30);
P_train=pixel_value(train_label,:)';
dtrain_label=train_label-floor(train_label/N)*N;
dtrain_label(dtrain_label==0)=N;
T_train=direction_label(dtrain_label,:)';
% 测试集
test_label=rand_label(31:end);
P_test=pixel_value(test_label,:)';
dtest_label=test_label-floor(test_label/N)*N;
dtest_label(dtest_label==0)=N;
T_test=direction_label(dtest_label,:)'
%% 创建BP网络
net=newff(minmax(P_train),[10,3],{'tansig','purelin'},'trainlm');
% 设置训练参数
net.trainParam.epochs=1000;
net.trainParam.show=10;
net.trainParam.goal=1e-3;
net.trainParam.lr=0.1;
%% 网络训练
net=train(net,P_train,T_train);
%% 仿真测试
T_sim=sim(net,P_test);
for i=1:3
    for j=1:20
        if T_sim(i,j)<0.5
            T_sim(i,j)=0;
        else
            T_sim(i,j)=1;
        end
    end
end
T_sim
T_test
web browser http://www.ilovematlab.cn/thread-61927-1-1.html
%%
% 
% <html>
% <table align="center" >	<tr>		<td align="center"><font size="2">版权所有：</font><a
% href="http://www.ilovematlab.cn/">Matlab中文论坛</a>&nbsp;&nbsp; <script
% src="http://s3.cnzz.com/stat.php?id=971931&web_id=971931&show=pic" language="JavaScript" ></script>&nbsp;</td>	</tr></table>
% </html>
% 
