%% SVM���������Ϣ����ʱ��ع�Ԥ��----��ָ֤������ָ���仯���ƺͱ仯�ռ�Ԥ�� 
%
% <html>
% <table border="0" width="600px" id="table1">	<tr>		<td><b><font size="2">�ð�������������</font></b></td>	</tr>	<tr><td><span class="comment"><font size="2">1�����˳���פ���ڴ�<a target="_blank" href="http://www.ilovematlab.cn/forum-158-1.html"><font color="#0000FF">���</font></a>��Ըð������ʣ��������ʱش𡣱����鼮�ٷ���վΪ��<a href="http://video.ourmatlab.com">video.ourmatlab.com</a></font></span></td></tr><tr>		<td><font size="2">2�����<a href="http://union.dangdang.com/transfer/transfer.aspx?from=P-284318&backurl=http://www.dangdang.com/">�ӵ���Ԥ������</a>��<a href="http://union.dangdang.com/transfer/transfer.aspx?from=P-284318&backurl=http://www.dangdang.com/">��Matlab������30������������</a>��</td></tr><tr>	<td><p class="comment"></font><font size="2">3</font><font size="2">���˰��������׵Ľ�ѧ��Ƶ����Ƶ���ط�ʽ<a href="http://video.ourmatlab.com/vbuy.html">video.ourmatlab.com/vbuy.html</a></font><font size="2">�� </font></p></td>	</tr>			<tr>		<td><span class="comment"><font size="2">		4���˰���Ϊԭ��������ת����ע����������Matlab������30����������������</font></span></td>	</tr>		<tr>		<td><span class="comment"><font size="2">		5�����˰��������������о��й��������ǻ�ӭ���������Ҫ��ȣ����ǿ��Ǻ���Լ��ڰ����</font></span></td>	</tr>		</table>
% </html>
% 
% by liyang[faruto] @ faruto's Studio~
% Email:faruto@163.com
% QQ:516667408 
% http://blog.sina.com.cn/faruto
% http://www.matlabsky.com
% http://www.mfun.la
% http://video.ourmatlab.com
%% ��ջ�������
function chapter15
tic;
close all;
clear;
clc;
format compact;
%% ԭʼ���ݵ���ȡ

% �������������ָ֤��(1990.12.19-2009.08.19)
% ������һ��4579*6��double�͵ľ���,ÿһ�б�ʾÿһ�����ָ֤��
% 6�зֱ��ʾ������ָ֤���Ŀ���ָ��,ָ�����ֵ,ָ�����ֵ,����ָ��,���ս�����,���ս��׶�.
load chapter15_sh.mat;

% ��ȡ����
ts = sh_open;
time = length(ts);

% ����ԭʼ��ָ֤����ÿ�տ�����
figure;
plot(ts,'LineWidth',2);
title('��ָ֤����ÿ�տ�����(1990.12.20-2009.08.19)','FontSize',12);
grid on;
snapnow;
%% ��ԭʼ���ݽ���ģ����Ϣ����

win_num = floor(time/5);
tsx = 1:win_num;
tsx = tsx';
[Low,R,Up]=FIG_D(ts','triangle',win_num);

% ģ����Ϣ�������ӻ�ͼ
figure;
hold on;
plot(Low,'b+');
plot(R,'r*');
plot(Up,'gx');
hold off;
legend('Low','R','Up',2);
title('ģ����Ϣ�������ӻ�ͼ','FontSize',12);
grid on;
snapnow;
%% ����SVM��Low���лع�Ԥ��

% ����Ԥ����,��Low���й�һ������
% mapminmaxΪmatlab�Դ���ӳ�亯��
[low,low_ps] = mapminmax(Low);
low_ps.ymin = 100;
low_ps.ymax = 500;
% ��Low���й�һ��
[low,low_ps] = mapminmax(Low,low_ps);
% ����Low��һ�����ͼ��
figure;
plot(low,'b+');
title('Low��һ�����ͼ��','FontSize',12);
grid on;
% ��low����ת��,�Է���libsvm����������ݸ�ʽҪ��
low = low';
snapnow;

% ѡ��ع�Ԥ���������ѵ�SVM����c&g
% ���Ƚ��д���ѡ��
[bestmse,bestc,bestg] = SVMcgForRegress(low,tsx,-10,10,-10,10,3,1,1,0.1);

% ��ӡ����ѡ����
disp('��ӡ����ѡ����');
str = sprintf( 'SVM parameters for Low:Best Cross Validation MSE = %g Best c = %g Best g = %g',bestmse,bestc,bestg);
disp(str);

% ���ݴ���ѡ��Ľ��ͼ�ٽ��о�ϸѡ��
[bestmse,bestc,bestg] = SVMcgForRegress(low,tsx,-4,8,-10,10,3,0.5,0.5,0.05);

% ��ӡ��ϸѡ����
disp('��ӡ��ϸѡ����');
str = sprintf( 'SVM parameters for Low:Best Cross Validation MSE = %g Best c = %g Best g = %g',bestmse,bestc,bestg);
disp(str);

% ѵ��SVM
cmd = ['-c ', num2str(bestc), ' -g ', num2str(bestg) , ' -s 3 -p 0.1'];
low_model = svmtrain(low, tsx, cmd);

% Ԥ��
[low_predict,low_mse] = svmpredict(low,tsx,low_model);
low_predict = mapminmax('reverse',low_predict,low_ps);
predict_low = svmpredict(1,win_num+1,low_model);
predict_low = mapminmax('reverse',predict_low,low_ps);
predict_low
%% ����Low�Ļع�Ԥ��������
figure;
hold on;
plot(Low,'b+');
plot(low_predict,'r*');
legend('original low','predict low',2);
title('original vs predict','FontSize',12);
grid on;
figure;
error = low_predict - Low';
plot(error,'ro');
title('���(predicted data-original data)','FontSize',12);
grid on;
snapnow;
%% ����SVM��R���лع�Ԥ��

% ����Ԥ����,��R���й�һ������
% mapminmaxΪmatlab�Դ���ӳ�亯��
[r,r_ps] = mapminmax(R);
r_ps.ymin = 100;
r_ps.ymax = 500;
% ��R���й�һ��
[r,r_ps] = mapminmax(R,r_ps);
% ����R��һ�����ͼ��
figure;
plot(r,'r*');
title('r��һ�����ͼ��','FontSize',12);
grid on;
% ��R����ת��,�Է���libsvm����������ݸ�ʽҪ��
r = r';
snapnow;

% ѡ��ع�Ԥ���������ѵ�SVM����c&g
% ���Ƚ��д���ѡ��
[bestmse,bestc,bestg] = SVMcgForRegress(r,tsx,-10,10,-10,10,3,1,1,0.1);

% ��ӡ����ѡ����
disp('��ӡ����ѡ����');
str = sprintf( 'SVM parameters for R:Best Cross Validation MSE = %g Best c = %g Best g = %g',bestmse,bestc,bestg);
disp(str);

% ���ݴ���ѡ��Ľ��ͼ�ٽ��о�ϸѡ��
[bestmse,bestc,bestg] = SVMcgForRegress(r,tsx,-4,8,-10,10,3,0.5,0.5,0.05);

% ��ӡ��ϸѡ����
disp('��ӡ��ϸѡ����');
str = sprintf( 'SVM parameters for R:Best Cross Validation MSE = %g Best c = %g Best g = %g',bestmse,bestc,bestg);
disp(str);

% ѵ��SVM
cmd = ['-c ', num2str(bestc), ' -g ', num2str(bestg) , ' -s 3 -p 0.1'];
r_model = svmtrain(r, tsx, cmd);

% Ԥ��
[r_predict,r_mse] = svmpredict(r,tsx,low_model);
r_predict = mapminmax('reverse',r_predict,r_ps);
predict_r = svmpredict(1,win_num+1,r_model);
predict_r = mapminmax('reverse',predict_r,r_ps);
predict_r
%% ����R�Ļع�Ԥ��������
figure;
hold on;
plot(R,'b+');
plot(r_predict,'r*');
legend('original r','predict r',2);
title('original vs predict','FontSize',12);
grid on;
figure;
error = r_predict - R';
plot(error,'ro');
title('���(predicted data-original data)','FontSize',12);
grid on;
snapnow;
%% ����SVM��Up���лع�Ԥ��

% ����Ԥ����,��up���й�һ������
% mapminmaxΪmatlab�Դ���ӳ�亯��
[up,up_ps] = mapminmax(Up);
up_ps.ymin = 100;
up_ps.ymax = 500;
% ��Up���й�һ��
[up,up_ps] = mapminmax(Up,up_ps);
% ����Up��һ�����ͼ��
figure;
plot(up,'gx');
title('Up��һ�����ͼ��','FontSize',12);
grid on;
% ��up����ת��,�Է���libsvm����������ݸ�ʽҪ��
up = up';
snapnow;

% ѡ��ع�Ԥ���������ѵ�SVM����c&g
% ���Ƚ��д���ѡ��
[bestmse,bestc,bestg] = SVMcgForRegress(up,tsx,-10,10,-10,10,3,1,1,0.5);

% ��ӡ����ѡ����
disp('��ӡ����ѡ����');
str = sprintf( 'SVM parameters for Up:Best Cross Validation MSE = %g Best c = %g Best g = %g',bestmse,bestc,bestg);
disp(str);

% ���ݴ���ѡ��Ľ��ͼ�ٽ��о�ϸѡ��
[bestmse,bestc,bestg] = SVMcgForRegress(up,tsx,-4,8,-10,10,3,0.5,0.5,0.2);

% ��ӡ��ϸѡ����
disp('��ӡ��ϸѡ����');
str = sprintf( 'SVM parameters for Up:Best Cross Validation MSE = %g Best c = %g Best g = %g',bestmse,bestc,bestg);
disp(str);

% ѵ��SVM
cmd = ['-c ', num2str(bestc), ' -g ', num2str(bestg) , ' -s 3 -p 0.1'];
up_model = svmtrain(up, tsx, cmd);

% Ԥ��
[up_predict,up_mse] = svmpredict(up,tsx,up_model);
up_predict = mapminmax('reverse',up_predict,up_ps);
predict_up = svmpredict(1,win_num+1,up_model);
predict_up = mapminmax('reverse',predict_up,up_ps);
predict_up
%% ����Up�Ļع�Ԥ��������
figure;
hold on;
plot(Up,'b+');
plot(up_predict,'r*');
legend('original up','predict up',2);
title('original vs predict','FontSize',12);
grid on;
figure;
error = up_predict - Up';
plot(error,'ro');
title('���(predicted data-original data)','FontSize',12);
grid on;
toc;
snapnow;
% web http://www.matlabsky.com/forum-31-1.html
web http://www.matlabsky.com/forum-31-1.html -new;
%% �Ӻ��� SVMcgForRegress.m
function [mse,bestc,bestg] = SVMcgForRegress(train_label,train,cmin,cmax,gmin,gmax,v,cstep,gstep,msestep)
% SVMcgForClass
% ����:
% train_label:ѵ������ǩ.Ҫ����libsvm��������Ҫ��һ��.
% train:ѵ����.Ҫ����libsvm��������Ҫ��һ��.
% cmin:�ͷ�����c�ı仯��Χ����Сֵ(ȡ��2Ϊ�׵Ķ�����),�� c_min = 2^(cmin).Ĭ��Ϊ -5
% cmax:�ͷ�����c�ı仯��Χ�����ֵ(ȡ��2Ϊ�׵Ķ�����),�� c_max = 2^(cmax).Ĭ��Ϊ 5
% gmin:����g�ı仯��Χ����Сֵ(ȡ��2Ϊ�׵Ķ�����),�� g_min = 2^(gmin).Ĭ��Ϊ -5
% gmax:����g�ı仯��Χ����Сֵ(ȡ��2Ϊ�׵Ķ�����),�� g_min = 2^(gmax).Ĭ��Ϊ 5
% v:cross validation�Ĳ���,�������Լ���Ϊ�����ֽ���cross validation.Ĭ��Ϊ 3
% cstep:����c�����Ĵ�С.Ĭ��Ϊ 1
% gstep:����g�����Ĵ�С.Ĭ��Ϊ 1
% msestep:�����ʾMSEͼʱ�Ĳ�����С.Ĭ��Ϊ 20
% ���:
% bestacc:Cross Validation �����е���߷���׼ȷ��
% bestc:��ѵĲ���c
% bestg:��ѵĲ���g

% about the parameters of SVMcgForRegress
if nargin < 10
    msestep = 0.1;
end
if nargin < 7
    msestep = 0.1;
    v = 3;
    cstep = 1;
    gstep = 1;
end
if nargin < 6
    msestep = 0.1;
    v = 3;
    cstep = 1;
    gstep = 1;
    gmax = 5;
end
if nargin < 5
    msestep = 0.1;
    v = 3;
    cstep = 1;
    gstep = 1;
    gmax = 5;
    gmin = -5;
end
if nargin < 4
    msestep = 0.1;
    v = 3;
    cstep = 1;
    gstep = 1;
    gmax = 5;
    gmin = -5;
    cmax = 5;
end
if nargin < 3
    msestep = 0.1;
    v = 3;
    cstep = 1;
    gstep = 1;
    gmax = 5;
    gmin = -5;
    cmax = 5;
    cmin = -5;
end
% X:c Y:g cg:mse
[X,Y] = meshgrid(cmin:cstep:cmax,gmin:gstep:gmax);
[m,n] = size(X);
cg = zeros(m,n);
% record accuracy with different c & g,and find the best mse with the smallest c
bestc = 0;
bestg = 0;
mse = 10^10;
basenum = 2;
for i = 1:m
    for j = 1:n
        cmd = ['-v ',num2str(v),' -c ',num2str( basenum^X(i,j) ),' -g ',num2str( basenum^Y(i,j) ),' -s 3'];
        cg(i,j) = svmtrain(train_label, train, cmd);
        
        if cg(i,j) < mse
            mse = cg(i,j);
            bestc = basenum^X(i,j);
            bestg = basenum^Y(i,j);
        end
        if ( cg(i,j) == mse && bestc > basenum^X(i,j) )
            mse = cg(i,j);
            bestc = basenum^X(i,j);
            bestg = basenum^Y(i,j);
        end
        
    end
end

[cg,ps] = mapminmax(cg);
% draw the accuracy with different c & g
figure;
[C,h] = contour(X,Y,cg,0:msestep:1);
clabel(C,h,'FontSize',10,'Color','r');
xlabel('log2c','FontSize',10);
ylabel('log2g','FontSize',10);
grid on;

%%%
% 
% <html>
% <table width="656" align="left" >	<tr><td align="center"><p><font size="2"><a href="http://video.ourmatlab.com/">Matlab������30����������</a></font></p><p align="left"><font size="2">�����̳��</font></p><p align="left"><font size="2">��Matlab������30�������������ٷ���վ��<a href="http://video.ourmatlab.com">video.ourmatlab.com</a></font></p><p align="left"><font size="2">Matlab������̳��<a href="http://www.matlabsky.com">www.matlabsky.com</a></font></p><p align="left"><font size="2">M</font><font size="2">atlab�����ٿƣ�<a href="http://www.mfun.la">www.mfun.la</a></font></p><p align="left"><font size="2">Matlab������̳��<a href="http://www.ilovematlab.com">www.ilovematlab.com</a></font></p></td>	</tr></table>
% </html>
% 

