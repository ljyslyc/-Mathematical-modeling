function I = getPicData()
% getPicData.m
% ��ȡdigital_picĿ¼�µ�����ͼ��
% output:
% I : 64 * 64 * 1000, ����1000��64*64��ֵͼ��

I = zeros(64,64,1000);
k = 1;

% ���ѭ������ȡ��ͬ���ֵ�ͼ��
for i=1:10
    % �ڲ�ѭ���� ��ȡͬһ���ֵ�100��ͼ
    for j=1:100
        file = sprintf('digital_pic\\%d_%03d.bmp', i-1, j);
        I(:,:,k) = imread(file);
        
        % ͼ�������
        k = k + 1;
    end
end
