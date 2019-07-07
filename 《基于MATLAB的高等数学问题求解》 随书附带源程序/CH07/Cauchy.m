function xi=Cauchy(fx,Fx,range)                                 
%CAUCHY   ��֤������ĳ���������Ƿ����������ֵ����                                  
% CAUCHY(F,G,RANGE)  ��ͼ�εķ�ʽ��ʾ������ĳ�������ϵĿ�����ֵ����                    
% XI=CAUCHY(F,G,RANGE)  ���غ�����ָ�������ϵ�һ��������ֵ��                      
%                                                               
% ���������                                                         
%     ---F,G��������MATLAB��������������������������������M�ļ�                       
%     ---RANGE��ָ��������                                            
% ���������                                                         
%     ---XI��������ֵ��                                               
%                                                               
% Sea also Rolle, Lagange                                       
                                                                
fab=subs(fx,range);                                             
Fab=subs(Fx,range);                                             
df=diff(fx);                                                    
dF=diff(Fx);                                                    
while 1                                                         
    x=fzero(inline(df/dF-diff(fab)/diff(Fab)),rand);            
    if prod(subs(Fx,x)-range)<=0                                
        break                                                   
    end                                                         
end                                                             
if nargout==1                                                   
    xi=x;                                                       
else                                                            
    ezplot(Fx,fx,range)                                         
    hold on                                                     
    x_range=[subs(Fx,x)-diff(Fab)/10,subs(Fx,x)+diff(Fab)/10];  
    y_range=diff(fab)/diff(Fab)*(x_range-subs(Fx,x))+subs(fx,x);
    plot(x_range,y_range,'k--')                                 
    title(['\xi=',num2str(x)])                                  
end                                                             
web -broswer http://www.ilovematlab.cn/forum-221-1.html