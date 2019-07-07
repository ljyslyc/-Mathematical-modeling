function c=sym_poly(s,v,flag)                                
%SYM_POLY   ���Ŷ���ʽ�����ʽϵ��֮����໥ת��                               
% C=SYM_POLY(S,V,1)��C=SYM_POLY(S,V,'sym2poly')  ��ȡ���Ŷ���ʽS��ϵ������
% S=SYM_POLY(C,V,2)��S=SYM_POLY(C,V,'poly2sym')  ��ϵ������C�������Ŷ���ʽ
%                                                            
% ���������                                                      
%     ---S������ķ��Ŷ���ʽ                                          
%     ---V�����Ŷ���ʽ�ķ����Ա���                                       
%     ---C�������ϵ������                                           
%     ---FLAG��ָ��ת�����򣬵�FLAG=1��'sym2poly'��ʾ�ɶ���ʽ��ϵ������ת����        
%                ��FLAG=2��'poly2sym'��ʾ��ϵ�����������ʽת��             
% ���������                                                      
%     ---C�����صķ��Ŷ���ʽ��ϵ������                                     
%     ---S����ϵ�����������õ��ķ��Ŷ���ʽ                                   
%                                                            
% See also poly2sym, sym2poly                                
                                                             
k=1;                                                         
switch flag                                                  
    case {1,'sym2poly'}                                      
        c=subs(s,v,0);                                       
        while 1                                              
            ds=diff(s,v);                                    
            c=[subs(ds,v,0)/prod(1:k),c];                    
            s=ds;                                            
            if ~ismember(sym(v),symvar(ds))                  
                break                                        
            end                                              
            k=k+1;                                           
        end                                                  
    case {2,'poly2sym'}                                      
        n=length(s);                                         
        c=s*(sym(v)).^(n-1:-1:0).';                          
    otherwise                                                
        error('Error flag.')                                 
end                                                          
web -broswer http://www.ilovematlab.cn/forum-221-1.html