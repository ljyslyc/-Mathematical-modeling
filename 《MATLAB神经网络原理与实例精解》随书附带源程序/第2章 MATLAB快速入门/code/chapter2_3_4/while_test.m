% while_test.m
i=1;
while i<=5,			
    s(i)=1;
    j=1;
   while j<=i			%�ڲ�ѭ��������׳�
       s(i)=s(i)*j;
       j=j+1;
   end
   i=i+1;
end
s
