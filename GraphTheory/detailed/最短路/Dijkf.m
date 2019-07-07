function [d path] = Dijkf(W,start)
% Dijkstra�㷨����ȡ��ʼ�㵽�������е�����·��
% input arg:W��ͼ��Ȩֵ����start�ǳ�ʼ��ı�ţ���1��ʼ
% d��ʾ��̾����ֵ��path��·������

% ������ʼ��
path=zeros(length(W),length(W));
path(:,1)=start;
d(1:length(W)) = W(start,:);
d(start)=0;
visited(1:length(W)) =0;
visited(start) =1;
while sum(visited) < length(W)
%     �ҵ�δ���ʵĽڵ�
    tb = find(visited == 0);
%     �ҵ�δ���ʽڵ��о����ʼ�������̵Ľڵ�
    tmpb = find(d(tb) == min(d(tb)));
    tmpb = tb(tmpb(1));
%     ���Ϊ�ѷ���״̬
    visited(tmpb) = 1;
%     ���������Ľڵ�
    tb = find(visited == 0);
    tx = find(d(tb) > d(tmpb) + W(tmpb,tb));
    tb = tb(tx);
    d(tb) = d(tmpb) + W(tmpb,tb); 
%     ����pathֵ
     for i = 1:length(tb)
        path(tb(i),:) = path(tmpb,:);
        x = find(path(tb(i),:)==0);
        path(tb(i),x(1))=tmpb;
        path(tb(i),x(2))=tb(i);
     end
end
% �����̽���
end

