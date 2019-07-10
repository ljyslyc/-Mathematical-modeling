function [flag,dis]=Bellmanford(G,s)
% ������Ƿ���ڿ��н�
    %G��ͼ���ڽӾ����ʾ��Ԫ��ֵΪȨ��
    %s��ͼ��Դ��
    dis = ones(1,size(G,1))*inf;
    %��ʼ��
    dis = init(G,s,dis);
    %ִ���ɳڲ���
    for l=1:size(G,1)-1
        for j=1:size(G,1)
            for k=1:size(G,1)
                dis = relax(G,j,k,dis);
            end
        end
    end
    %�ж��Ƿ����Ȩ��Ϊ��ֵ�Ļ�·
    for m=1:size(G,1)
        for n=1:size(G,1)
            %�Ƿ���ڹ��ƴ��������������ڣ���������Ȩ��Ϊ��ֵ�Ļ�
            if dis(n)>dis(m) + G(m,n)
                flag = 0;
                return;
            end
        end
    end
    flag = 1;
end

%dis�����·������ֵ����
%G��ͼ���ڽӱ��ʾ����Ԫ�ش�������i������j֮��ߵ�Ȩ��
function [dis] = init(G,s,dis)
    for i=1:size(G,1)
        dis(i) = inf;
    end
    dis(s) = 0;%Դ��ľ���Ϊ0
end

%dis�����·������ֵ����
%G��ͼ���ڽӱ��ʾ����Ԫ�ش�������i������j֮��ߵ�Ȩ��
function [dis] = relax(G,u,v,dis)
    %dis(v):��ʾG�д�Դ�㵽��v�ľ������ֵ��������ֵ����ǰ���ڵ�ľ���+u��v�ľ��룬�����
	if dis(v)>dis(u)+G(u,v)
        dis(v) = dis(u)+G(u,v);
 	end
end