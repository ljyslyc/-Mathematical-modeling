function S = summation(n)                          
%SUMMATION   ���ʽ1+2*3+3*4+4*5+...                  
% S=SUMMATION(N)  ���õݹ��㷨���1+2*3+3*4+4*5+...+N*(N+1)
%                                                  
% ���������                                            
%     ---N������                                      
% ���������                                            
%     ---S����ʽ�ĺ�                                    
%                                                  
% See also sum, prod                               
                                                   
if n==1                                            
    S=1;                                           
else                                               
    S=n*(n+1)+summation(n-1);                      
end                                                
web -broswer http://www.ilovematlab.cn/forum-221-1.html