disp('==�жϳ���A�Ƿ��Ǻ���f(x)��x0��������==')       
A=input('��������ܼ���ֵA=');                   
x0=input('�����뼫�޵�x0=');                   
f=input('�����뼫�ޱ��ʽf(x)=','s');            
n=1;flag=1;delta=1;                      
x=x0-delta;                              
while flag==1                            
    epsilon=input('��������һ������С������=');      
    while abs(eval(f)-A)>epsilon         
        delta=delta/2;                   
        x=x0-delta;                      
        if abs(delta)<eps                
            disp('�Ҳ�����')                 
            n=0;break                    
        end                              
    end                                  
    if n==0                              
        disp('���޲���ȷ')                   
        break                            
    end                                  
    disp(['��=',num2str(delta)])          
    flag=input('Ҫ����һ���������԰�1�������������ּ���');
end                                      
web -broswer http://www.ilovematlab.cn/forum-221-1.html