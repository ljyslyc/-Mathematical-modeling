% example4_6_1.m 
%% ����
clear,clc
close all

%% adapt���ڸ�֪��

% ������֪��
net=newp([-1,2;-2,2],1);

% ����ѵ������
P={[0;0] [0;1] [1;0] [1;1]};
T={0,0,1,1};

% ���е���
[net,y,ee,pf] = adapt(net,P,T);
ma=mae(ee)
ite=0;
while ma>=0.15
  [net,y,ee,pf] = adapt(net,P,T,pf);  
  ma=mae(ee)
  newT=sim(net,P)
  ite=ite+1;
  if ite>=10
      break;
  end
end
web -broswer http://www.ilovematlab.cn/forum-222-1.html