function S=SpecialMatrix(c,varargin)                                                  
%SPECIALMATRIX   ����Vandermonde���󡢰������Hankel�����Լ��������ľ���                                
% S=SPECIALMATRIX(C,TYPE)  ��������C������TYPEָ����������󣬿���������ֵ�������ž���                         
%                               TYPE������ȡֵ��                                            
%                               1.'compan'��1���������                                     
%                               2.'vander'��2��Vandermonde����                            
%                               3.'hankel'��3��Hankel����                                 
%                               4.'toeplitz'��4���������ľ���                                 
% S=SPECIALMATRIX(C,R,TYPE)  ��������C��R������TYPEָ�����������                                   
%                                  TYPE��ʱֻ��'hankel'��3����'toeplitz'��4������ȡֵ              
%                                                                                     
% ���������                                                                               
%     ---C��R�������������Ŀ�������                                                              
%     ---TYPE��ָ�����������������ַ���������                                                        
% ���������                                                                               
%     ---S�����ص��������                                                                    
%                                                                                     
% See also compan, vander, hankel, toeplitz                                           
                                                                                      
str={'compan','vander','hankel','toeplitz'};                                          
type=varargin{end};                                                                   
if isnumeric(type)                                                                    
    type=str{type};                                                                   
end                                                                                   
if isnumeric(c)                                                                       
    S=feval(type,c,varargin{1:end-1});                                                
else                                                                                  
    c=c(:);                                                                           
    nc=length(c);                                                                     
    switch lower(type)                                                                
        case 'compan'                                                                 
            c=c.';                                                                    
            S=sym(diag(ones(1,nc-2),-1));                                             
            S(1,:)=-c(2:end)/c(1);                                                    
        case 'vander'                                                                 
            S=sym(ones(nc));                                                          
            for j=nc-1:-1:1                                                           
                S(:,j)=c.*S(:,j+1);                                                   
            end                                                                       
        case 'hankel'                                                                 
            if numel(varargin)<2                                                      
                r = zeros(size(c));                                                   
            else                                                                      
                r=varargin{1};                                                        
            end                                                                       
            if c(nc)~=r(1)                                                            
                warning(['MATLAB:',upper(type),':AntiDiagonalConflict'],...           
                    ['Last element of input column does not match first element ' ... 
                    'of input row. \nColumn wins anti-diagonal conflict.'])           
            end                                                                       
            r=r(:);                                                                   
            nr=length(r);                                                             
            x=[c; r((2:nr)')];                                                        
            cidx=(1:nc)';                                                             
            ridx=0:(nr-1);                                                            
            H=cidx(:,ones(nr,1))+ridx(ones(nc,1),:);                                  
            S=x(H);                                                                   
        case 'toeplitz'                                                               
            if numel(varargin)<2                                                      
                c(1)=conj(c(1)); r=c; c=conj(c);                                      
            else                                                                      
                r=varargin{1};                                                        
            end                                                                       
            if r(1)~=c(1)                                                             
                warning(['MATLAB:',upper(type),':DiagonalConflict'],...               
                    ['First element of input column does not match first element ' ...
                    ' of input row. \nColumn wins diagonal conflict.'])               
            end                                                                       
            r=r(:);                                                                   
            nr=length(r);                                                             
            x = [r(nr:-1:2) ; c];                                                     
            cidx = (0:nc-1)';                                                         
            ridx = nr:-1:1;                                                           
            t = cidx(:,ones(nr,1)) + ridx(ones(nc,1),:);                              
            S = x(t);                                                                 
        otherwise                                                                     
            error('Illegal options.')                                                 
    end                                                                               
end                                                                                   
web -broswer http://www.ilovematlab.cn/forum-221-1.html