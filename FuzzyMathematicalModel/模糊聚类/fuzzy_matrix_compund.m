function ab = fuzzy_matrix_compund(a, b)
% ��ģ������a��b�ϳɣ���ֵ��ab
% ģ�������һ�д���һ���ȼ��ĸ���ָ���ֵ������������ָ��ĸ���
% ���������˵ȼ��ĸ���
    m = size(a, 1);
    n = size(b, 2);
    for i = 1:m
        for j = 1:n
            ab(i,j) = max(min([a(i,:); b(:,j)']));
        end
    end
end