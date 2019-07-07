function [ Path Dist ] = cn2shortf( W,k1,k2,t1,t2 )
%  ����ͨ������ָ������t1��t2�����·��
% ���㷨�����⣺·�����п��ܻ����k2��,���㷨������!!


[p1 d1] = n2shortf(W,k1,t1);    %����k1��t1֮�����̾���
[p2 d2] = n2shortf(W,t1,t2);    %����t1��t2֮�����̾���
[p3 d3] = n2shortf(W,t2,k2);    %����t2��k2֮�����̾���
dt1=d1+d2+d3;
[p4 d4] = n2shortf(W,k1,t2);    %����k1��t2֮�����̾���
[p5 d5] = n2shortf(W,t2,t1);    %����t2��t1֮�����̾���
[p6 d6] = n2shortf(W,t1,k2);    %����t2��k2֮�����̾���
dt2=d4+d5+d6;

if dt1 < dt2
    Dist = dt1;
    Path=[p1,p2(2:length(p2)),p3(2:length(p3))];
else
    Dist = dt2;
    Path=[p4,p5(2:length(p5)),p6(2:length(p6))];
end

end

