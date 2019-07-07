function [AX,H1,H2]=plotxy(varargin)                                   
%PLOTXY   ����˫����ϵ                                                       
% PLOTXY(X1,Y1,X2,Y2)  ��˫����ϵ��ʹ��plot�����ֱ�������ݶ�X1-Y1��X2-Y2��ͼ��             
% PLOTXY(X1,Y1,X2,Y2,FUN)  ��˫����ϵ��ʹ�û�ͼ����FUN�ֱ�������ݶ�X1-Y1��X2-Y2��          
%                               ͼ�Σ�FUN������MATLAB�Դ���ͼ����������plot��semilogx�� 
%                               loglog��stem�ȣ�Ҳ�������û��Ա��ͼ�����������Աຯ������     
%                               ��������h=function(x,y)�ĵ��ø�ʽ               
% PLOTXY(X1,Y1,X2,Y2,FUN1,FUN2)  ��˫����ϵ��ʹ�û�ͼ����FUN1��FUN2�ֱ�������ݶ�X1-Y1     
%                                      ��X2-Y2��ͼ�Σ�����FUN1��FUN2�ĸ�ʽ��FUN��ȫһ��
% AX=PLOTXY(...)  ����˫����ϵͼ�β�����˫����ϵ��������������                              
% [AX,H1,H2]=PLOTXY(...)  ����˫����ϵͼ�β��������������������������е�ͼ�ζ�����               
%                                                                      
% ���������                                                                
%     ---X1,Y1,X2,Y2����ͼ����                                              
%     ---FUN,FUN1,FUN2��ָ���Ļ�ͼ����                                         
% ���������                                                                
%     ---AX��������������                                                    
%     ---H1,H2��������ϵ�е�ͼ�ζ�����                                            
%                                                                      
% See also PLOTYY                                                      
%                                                                      
args=varargin;                                                         
[x1,y1,x2,y2]=deal(args{1:4});                                         
if nargin<5, fun1 = @plot; else fun1 = args{5}; end                    
if nargin<6, fun2 = fun1;  else fun2 = args{6}; end                    
set(gcf,'NextPlot','add')                                              
hAxes1=axes;                                                           
h1=feval(fun1,hAxes1,x1,y1,'Color','b');                               
set(hAxes1,'XColor','b','YColor','b','Box','off')                      
hAxes2=axes('Position',get(hAxes1,'Position'));                        
h2=feval(fun2,hAxes2,x2,y2,'Color','r');                               
set(hAxes2,'Color','none','XColor','r','YColor','r','Box','off',...    
    'XAxisLocation','top','YAxisLocation','right')                     
if nargout==1                                                          
    AX=[hAxes1,hAxes2];                                                
elseif nargout==3                                                      
    AX=[hAxes1,hAxes2]; H1=h1; H2=h2;                                  
end                                                                    
web -broswer http://www.ilovematlab.cn/forum-221-1.html