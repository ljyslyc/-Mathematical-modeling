function hanoi(n,A,B,C)                              
%HANOI   ��������                                        
% HANOI(N,'A','B','C')  �ݹ��㷨����������                    
%                                                    
% ���������                                              
%     ---N���̵ĸ���                                      
%     ---'A','B','C'��������������                          
                                                     
fprintf('%d�����ӵ��ƶ����裺\n',n)                           
count=1;                                             
move(n,A,B,C)                                        
    function move(n,A,B,C)                           
        if n==0                                      
            return                                   
        else                                         
            move(n-1,A,C,B)                          
            disp(['��',int2str(count),'����',A,'-->',C])
            count=count+1;                           
            move(n-1,B,A,C)                          
        end                                          
    end                                              
end                                                  
web -broswer http://www.ilovematlab.cn/forum-221-1.html