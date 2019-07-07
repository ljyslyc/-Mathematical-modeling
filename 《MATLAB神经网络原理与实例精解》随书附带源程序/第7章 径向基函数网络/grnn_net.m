function y = grnn_net(p,t,x,spread)
% grnn_net.m
% p��R*Q���󣬰���Q������ΪR������
% t��1*Q����������Ӧ��p���������
% spread��ƽ�����ӣ���ѡ��ȱʡֵΪ0.5

if ~exist('spread','var')
    spread=0.5;
end

[R,Q]=size(p);
[R,S]=size(x);

%% �����������
yr = zeros(Q,S);
for i=1:S
    for j=1:Q
        v = norm(x(:,i) - p(:,j));
        yr(j,i) = exp(-v.^2/(2*spread.^2));
    end
end

%% �ӺͲ�����
ya = zeros(2,S);
for i=1:S
    ya(1,i) = sum(t .* yr(:,i)',2);
    ya(2,i) = sum(yr(:,i));
end

%% �����Ľ��
y = ya(1,:)./(eps+ya(2,:));
