% sa_tsp.m
% ��ģ���˻��㷨���TSP����

%% ����
close all
clear,clc

%% ��������,position��2��25�еľ���
position = [1304,2312;3639,1315;4177,2244;3712,1399;3488,1535;3326,1556;...
    3238,1229;4196,1044;4312,790;4386,570;3007,1970;2562,1756;...
    2788,1491;2381,1676;1322,695;3715,1678;3918,2179;4061,2370;...
    3394,2643;3439,3201;2935,3240;3140,3550;2545,2357;2778,2826;2360,2975]';
L = length(position);

% �����ڽӾ���dist  25*25
dist = zeros(L,L);
for i=1:L
   for j=1:L
       if i==j
           continue;
       end
      dist(i,j) = sqrt((position(1,i)-position(1,j)).^2 + (position(2,i)-position(2,j)).^2);
      dist(j,i) = dist(i,j);
   end
end

tic
%% ��ʼ��
MAX_ITER = 2000;
MAX_M = 20;
lambda = 0.97;
T0 = 100;
rng(2);
x0 = randperm(L);

%% 
T=T0;
iter = 1;
x=x0;                   % ·������
xx=x0;                  % ÿ��·��
di=tsp_len(dist, x0);   % ÿ��·����Ӧ�ľ���
n = 1;                  % ·������
% ��ѭ��
while iter <=MAX_ITER,
    
    % ��ѭ��������
    m = 1;
    % ��ѭ��
    while m <= MAX_M
        % ������·��
        newx = tsp_new_path(x);
        
        % �������
        oldl = tsp_len(dist,x);
        newl = tsp_len(dist,newx);
        if ( oldl > newl)   % �����·������ԭ·����ѡ����·����Ϊ��һ״̬
            x=newx;
            xx(n+1,:)=x;
            di(n+1)=newl;
            n = n+1;
            
        else                % �����·����ԭ·�����ִ�и��ʲ���
            tmp = rand;
            if tmp < exp(-(newl - oldl)/T)
                x=newx;
                xx(n+1,:)=x;
                di(n+1)=newl;
                n = n+1;
            end
        end
        m = m+1;            % ��ѭ��������1
    end                     % ��ѭ��
    iter = iter+1;          % ��ѭ��������1
    T = T*lambda;           % ����
end
toc

%% ��������ֵ
[bestd,index] = min(di);
bestx = xx(index,:);
fprintf('��ѡ�� %d ��·��\n', n);
fprintf('���Ž�:\n');
disp(bestd);
fprintf('����·��:\n');
disp(bestx);

%% ��ʾ
% ��ʾ·��ͼ
figure;
plot(position(1,:), position(2,:),'o');
hold on;
for i=1:L-1
   plot(position(1,bestx(i:i+1)), position(2,bestx(i:i+1))); 
end
plot([position(1,bestx(L)),position(1,bestx(1))], [position(2,bestx(L)),position(2,bestx(1))]); 
title('TSP����ѡ�������·��');
hold off;

% ��ʾ��ѡ���·���仯����
figure;
semilogx(1:n,di);
title('·�����ȵı仯����');


