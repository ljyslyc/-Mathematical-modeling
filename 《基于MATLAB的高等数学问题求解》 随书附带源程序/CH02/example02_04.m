web -broswer http://www.ilovematlab.cn/forum-221-1.html
month=input('�������·�month=');  % ��ʾ�����·�            
if month>12 || month<1 || mod(month,1)~=0  % �ж�����
    error('������·ݱ�����1~12��������')  % ���������������������ʾ    
end                                              
switch month                                     
    case {3 4 5}                                 
        season='spring';                         
    case {6 7 8}                                 
        season='summer';                         
    case {9 10 11}                               
        season='autumn';                         
    otherwise                                    
        season='winter';                         
end                                              
