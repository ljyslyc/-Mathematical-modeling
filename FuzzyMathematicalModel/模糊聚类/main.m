%% ģ���������֮�󣬸����������
load data1

% ����fuzzy_cluster_analysis.m֮��ķ�����
ind1 = [1,5];
ind2 = [2:3,6,8:11];
ind3 = [4,7];

so = [];
% ���������и�ѡ��һ��ȥ����ѭ������ÿһ��ȥ���ķ������������ƽ���ͣ��ҵ���С���Ǹ�����
for i = 1:length(ind1)
    for j = 1:length(ind3)
        for k = 1:length(ind2)
            t = [ind1(i), ind3(j), ind2(k)];
            err = caculate_SSE(A, t);
            so = [so;[t,err]];
        end
    end
end

so
tm = find(so(:,4) == min(so(:,4)));

result = so(tm,1:3)
