% elman_stockTest.m
%% ����
close all
clear,clc

%% ��������
load stock_net
load stock2
stock1=stock2';
% load stock1

% whos
%   Name        Size             Bytes  Class      Attributes
% 
%   net         1x1              71177  network              
%   stock1      1x280             2240  double    

% ��һ������
mi=min(stock1);
ma=max(stock1);
testdata = stock1(141:280);
testdata=(testdata-mi)/(ma-mi);

%% �ú�140������������
% ����
Pt=[];
for i=1:135
    Pt=[Pt;testdata(i:i+4)];
end
Pt=Pt';
% ����
Yt=sim(net,Pt); 

%���ݹ�һ����ʽ��Ԥ�����ݻ�ԭ�ɹ�Ʊ�۸�
YYt=Yt*(ma-mi)+mi;

%Ŀ������-Ԥ������
figure
plot(146:280, stock1(146:280), 'r',146:280, YYt, 'b');
legend('��ʵֵ', '���Խ��');
title('�ɼ�Ԥ�����');

%% 
%compute the Hit Rate
% count = 0;
% for i = 100:275
%     if (Store(i)-Store(i-1))*(YYt(i)-YYt(i-1))>0
%         count = count+1;
%     end
% end
% hit_rate=count/175
% 
% xlabel('Dates from 2008.06.16 to 2008.08.19(about the last 180days)');
% ylabel('Price');
% title('Simulation Datas Analysis---One day prediction')
% grid on
