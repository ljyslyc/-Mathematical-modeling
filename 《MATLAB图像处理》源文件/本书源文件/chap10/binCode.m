function s=binCode(a)
%�����������Ķ�������
if a>=0
    s=dec2bin(a);
else
%��a�ķ��룬���ء�01���ַ�������λȡ��
    s=dec2bin(abs(a));
    for t=1:numel(s)
        if s(t)=='0'
            s(t)='1';
        else s(t)='0';
        end
    end
end
