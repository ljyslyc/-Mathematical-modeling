%% ���������һ�����Ͱ���
% �Ա�������R�;��࣬����������Q�;���

% ��ָ�꿴�ɱ���������R�;���
load edu.txt
data = edu;
r = corrcoef(data);   % �������ϵ������
[m, n] = size(data);  % mΪ����������nΪָ�����
d = tril(r);         % ȡ��������Ԫ��
for i = 1:n
    d(i,i) = 0;
end

d = d(:);
d = nonzeros(d);     % ȡ������Ԫ��
d = d';
d = 1 -d;
z = linkage(d)
dendrogram(z)
% ������Ҫ����n��ָ������ѡ����������һ�ֵķ���

% ����ѡȡ6��ָ�꣬��������m����������Q�;���
data(:,3:6) = [];    % ����ѡȡ��ָ���ǵ�1,2,7,8,9,10���ɾ���м�������Ϣ
data = zscore(data);
y = pdist(data);
z = linkage(y)
dendrogram(z, 'average')