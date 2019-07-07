function [f] = boundnetf(G1,G2)
% ���������½�Ŀ�����
% G1��ʾ���������½����
% G2��ʾ���������½����
% f�����ж�ԭͼ���Ƿ���ڿ�����

n = size(G2,1);
G = G2 - G1;
% G(G~=0)=1;
x = zeros(1,n);
W = [0 x 0;
    x' G x';
    0 x 0;
    ];
for i = 1:n
    W(i+1,n+2)=sum(G1(i,:));
    W(1,i+1)=sum(G1(:,i));
end
     W(2,n+1) = inf;
     W(n+1,2) = inf;
  [f1 wf]=fofuf(W);
    if wf ~= sum(sum(G1))
        f = 0;
    else
        f1(2,n+1) = 0;
        f1(n+1,2) = 0;
        [f1,wf]=fofuf(G);
        f = f1;
    end
end

