function len = tsp_len(dis,path)
% dis:N*N�ڽӾ���
% ����ΪN��������������1-N������

N = length(path);
len = 0;
for i=1:N-1
    len = len + dis(path(i), path(i+1));
    
end

len = len + dis(path(1), path(N));

