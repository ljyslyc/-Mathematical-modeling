web -broswer http://www.ilovematlab.cn/forum-221-1.html
x=input('x=');  y=input('y=');  % ��ʾ����x��y
if x==0 | y==0  % �������ϵ�����                
    f=0;                                 
elseif x<0 & y<0  % ��������                 
    f=x^2*y;                             
elseif x<0 & y>0  % �ڶ�����                 
    f=x*y^2;                             
elseif x>0 & y<0  % ��������                 
    f=x^2*y^2;                           
else             % ����                    
    f=x^2*y^3;                           
end                                      
