% for_test.m
for i=1:5
   x=0;
    for j=1:i
       x=x+j; 
    end
    fprintf('��1��%d��������Ϊ ',i);
    disp(x)
end