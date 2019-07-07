%% SVM������Ļع�Ԥ�����---��ָ֤������ָ��Ԥ�� 
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
function chapter14
close all;
clear;
clc;
format compact;
%% ���ݵ���ȡ��Ԥ����

% �������������ָ֤��(1990.12.19-2009.08.19)
% ������һ��4579*6��double�͵ľ���,ÿһ�б�ʾÿһ�����ָ֤��
% 6�зֱ��ʾ������ָ֤���Ŀ���ָ��,ָ�����ֵ,ָ�����ֵ,����ָ��,���ս�����,���ս��׶�.
load chapter14_sh.mat;

% ��ȡ����
[m,n] = size(sh);
ts = sh(2:m,1);
tsx = sh(1:m-1,:);

% ����ԭʼ��ָ֤����ÿ�տ�����
figure;
plot(ts,'LineWidth',2);
title('��ָ֤����ÿ�տ�����(1990.12.20-2009.08.19)','FontSize',12);
grid on;

% ����Ԥ����,��ԭʼ���ݽ��й�һ��

ts = ts';
tsx = tsx';

% mapminmaxΪmatlab�Դ���ӳ�亯��
[TS,TSps] = mapminmax(ts);	
% ��ӳ�亯���ķ�Χ�����ֱ���Ϊ1��2
TSps.ymin = 1;
TSps.ymax = 2;
% ��ts���й�һ��
[TS,TSps] = mapminmax(ts,TSps);	

% ����ԭʼ��ָ֤����ÿ�տ�������һ�����ͼ��
figure;
plot(TS,'LineWidth',2);
title('ԭʼ��ָ֤����ÿ�տ�������һ�����ͼ��','FontSize',12);
grid on;
% ��TS����ת��,�Է���libsvm����������ݸ�ʽҪ��
TS = TS';

% mapminmaxΪmatlab�Դ���ӳ�亯��
[TSX,TSXps] = mapminmax(tsx);	
% ��ӳ�亯���ķ�Χ�����ֱ���Ϊ1��2
TSXps.ymin = 1;
TSXps.ymax = 2;
% ��tsx���й�һ��
[TSX,TSXps] = mapminmax(tsx,TSXps);	
% ��TSX����ת��,�Է���libsvm����������ݸ�ʽҪ��
TSX = TSX';
%% ѡ��ع�Ԥ�������ѵ�SVM����c&g

% ���Ƚ��д���ѡ��: 
% c �ı仯��Χ�� 2^(-5),2^(-4),...,2^(10)
% g �ı仯��Χ�� 2^(-5),2^(-4),...,2^(5)
[bestmse,bestc,bestg] = SVMcgForRegress(TS,TSX,-5,10,-5,5,3,1,1,0.0005);

% ��ӡ����ѡ����
disp('��ӡ����ѡ����');
str = sprintf( 'Best Cross Validation MSE = %g Best c = %g Best g = %g',bestmse,bestc,bestg);
disp(str);

% ���ݴ���ѡ��Ľ��ͼ�ٽ��о�ϸѡ��: 
% c �ı仯��Χ�� 2^(0),2^(0.3),...,2^(10)
% g �ı仯��Χ�� 2^(-2),2^(-1.7),...,2^(3)
[bestmse,bestc,bestg] = SVMcgForRegress(TS,TSX,0,10,-2,3,3,0.3,0.3,0.0002);

% ��ӡ��ϸѡ����
disp('��ӡ��ϸѡ����');
str = sprintf( 'Best Cross Validation MSE = %g Best c = %g Best g = %g',bestmse,bestc,bestg);
disp(str);

%% ���ûع�Ԥ�������ѵĲ�������SVM����ѵ��
cmd = ['-c ', num2str(bestc), ' -g ', num2str(bestg) , ' -s 3 -p 0.01'];
model = svmtrain(TS,TSX,cmd);

% model = svmtrain(TS,TSX,'-s 3 -c 1 -g 2 -p 0.01');

%% SVM����ع�Ԥ��
[predict,mse] = svmpredict(TS,TSX,model);
predict = mapminmax('reverse',predict,TSps);

% ��ӡ�ع���
str = sprintf( '������� MSE = %g ���ϵ�� R = %g%%',mse(2),mse(3)*100);
disp(str);
%% �������
figure;
hold on;
plot(ts,'LineWidth',2);
plot(predict,'r','LineWidth',2);
legend('ԭʼ����','�ع�Ԥ������');
hold off;
grid on;
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

