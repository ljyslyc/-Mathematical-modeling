% switch_test.m
value=input('������һ��0~9��������');
switch value,								%switch���
    case {1,3,5,7,9},
        fprintf('���������������!\n');
    case {0,2,4,6,8},
        fprintf('�����������ż��!\n');
    otherwise
        fprintf('����Ĳ���0~9������!\n');
end
