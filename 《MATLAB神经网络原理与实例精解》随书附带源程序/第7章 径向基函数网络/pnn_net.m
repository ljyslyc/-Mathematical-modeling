function clas = pnn_net(p,t,x,sigma)
% pnn_net.m
% pΪѵ������,R*Q��Q������ΪR������
% tΪѵ�����,1*Q��������ȡֵ��1~C����ʾC�����C<=Q
% xΪ��������,R*S����S������ΪR��������

% 
if ~exist('sigma','var')
    sigma=0.1;
end

% ���ݹ�һ��
MAX = max(p(:));
p=p/MAX;
x=x/MAX;

% ��������M����������N
[R,Q]=size(p);
[R,S]=size(x);

% ���㾶���������,y(i,j)�ǵ�j�������������i����Ԫ�����
y=zeros(Q,S);
for i=1:S
    for j=1:Q
        v = norm((x(:,i) - p(:,j))); %' * (x(:,i) - p(:,j));
        y(j,i) = exp(-v^2/(2*sigma^2));
    end
end

% ��Ӳ�
% ����C�����
C = length(unique(t));
% ��Ӳ����
vc=zeros(C,S);
for i=1:C
    for j=1:S
        vc(i,j) = mean(y(t==i,j));
    end
end

% �����
yout=zeros(C,S);
for i=1:S
    [~,index] = max(vc(:,i));
    yout(index,i) = 1;
end

clas=vec2ind(yout);
