function C=mat_add(varargin)                    
%MAT_ADD   ���������ͬά������ĺ�                         
% C=MAT_ADD(A,B,...)  ����������ĺ�                  
%                                               
% ���������                                         
%     ---A,B,...��ά����ͬ�ľ���                        
% ���������                                         
%     ---C�����صĺ;���                               
                                                
error(nargchk(2,inf,nargin))                    
C=varargin{1};                                  
s=size(C);                                      
for k=2:numel(varargin)                         
    B=varargin{k};                              
    s1=size(B);                                 
    if isequal(s,s1)                            
        C=C+B;                                  
    else                                        
        error('Martix dimension does''t match.')
    end                                         
end                                             
web -broswer http://www.ilovematlab.cn/forum-221-1.html