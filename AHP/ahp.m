function [maxeigval,w] = ahp(A)
    %AΪ�жϾ���
    [eigvec,eigval] = eig(A);
    eigval = diag(eigval);   %��������
    eigvalmag = imag(eigval);
    realind = find(eigvalmag < eps);
    realeigval = eigval(realind) %ʵ������
    maxeigval = max(realeigval) %�������ֵ
    index = find(eigval == maxeigval);
    vecinit = eigvec(:,index); %�������ֵ��Ӧ����������
    w = vecinit./sum(vecinit); %����������һ��
end