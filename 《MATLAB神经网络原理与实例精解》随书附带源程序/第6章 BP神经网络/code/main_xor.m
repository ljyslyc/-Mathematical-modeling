% �ű� main_xor.m
% ����ѵ����ʽ��BP����ʵ������߼���
% �������  ������

%% ����
clear all 
clc 
rng('default')
rng(0)

%% ����
eb = 0.01;                   % ������� 
eta = 0.6;                   % ѧϰ��
mc = 0.8;                    % �������� 
maxiter = 1000;              % ����������

%% ��ʼ������
nSampNum = 4; 
nSampDim = 2; 
nHidden = 3;    
nOut = 1; 
w = 2*(rand(nHidden,nSampDim)-1/2); 
b = 2*(rand(nHidden,1)-1/2); 
wex = [w,b]; 
 
W = 2*(rand(nOut,nHidden)-1/2); 
B = 2*(rand(nOut,1)-1/2); 
WEX = [W,B]; 

%% ����
SampIn=[0,0,1,1;...
        0,1,0,1;...
        1,1,1,1];
expected=[0,1,1,0];

%% ѵ��
iteration = 0; 
errRec = []; 
outRec = []; 
 
for i = 1 : maxiter    
    % �����ź����򴫲�
    hp = wex*SampIn;        
    tau = logsig(hp);      
    tauex  = [tau', 1*ones(nSampNum,1)]';    
     
    HM = WEX*tauex;    
    out = logsig(HM);   
    outRec = [outRec,out']; 
     
    err = expected - out; 
    sse = sumsqr(err);       
    errRec = [errRec,sse];
    fprintf('�� %d �ε������� %f \n',i,sse ) 
     
    % �ж��Ƿ�����
    iteration = iteration + 1;              
    if sse <= eb 
        break;
    end 
     
    % ����źŷ��򴫲�
    % DELTA��deltaΪ�ֲ��ݶ� 
    DELTA = err.*dlogsig(HM,out);            
    delta = W' * DELTA.*dlogsig(hp,tau);      
    dWEX = DELTA*tauex'; 
    dwex = delta*SampIn'; 
    
    % ����Ȩֵ
    if i == 1 
        WEX = WEX + eta * dWEX; 
        wex = wex + eta * dwex; 
    else    
        WEX = WEX + (1 - mc)*eta*dWEX + mc * dWEXOld; 
        wex = wex + (1 - mc)*eta*dwex + mc * dwexOld; 
    end 
    
    dWEXOld = dWEX; 
    dwexOld = dwex; 
   
    W  = WEX(:,1:nHidden); 
    
end      

%% ��ʾ

figure(1)
grid 
[nRow,nCol] = size(errRec); 
semilogy(1:nCol,errRec,'LineWidth',1.5); 
title('�������'); 
xlabel('��������'); 

x=-0.2:.05:1.2;
[xx,yy]=meshgrid(x);
for i=1:length(xx)
   for j=1:length(yy)
       xi=[xx(i,j),yy(i,j),1];
       hp = wex*xi';
       tau = logsig(hp);
       tauex  = [tau', 1]'; 
       HM = WEX*tauex;  
       out = logsig(HM); 
       z(i,j)=out;
   end
end
figure(2)
mesh(x,x,z);
figure(3)
plot([0,1],[0,1],'*','LineWidth',2);
hold on
plot([0,1],[1,0],'o','LineWidth',2);
[C,h]=contour(x,x,z,0.5,'b');
clabel(C,h);
legend('0','1','������');
title('������')

