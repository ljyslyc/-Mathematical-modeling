web -broswer http://www.ilovematlab.cn/forum-221-1.html
c=input('������һ���ַ�:','s');  % ��ʾ�����ַ�           
if c>='A' && c<='Z'                          
   disp(char(c+'a'-'A'));        % ���c��Ӧ��Сд�ַ�
elseif c>='a'&& c<='z'                       
    disp(char(c-'a'+'A'));   % ���c��Ӧ�Ĵ�д�ַ�    
elseif c>='0'&& c<='9'                       
    disp(c-'0');           % ���c��Ӧ������        
else                                         
    disp(c);                 % ���c           
end                                          
