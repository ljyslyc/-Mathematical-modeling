web -broswer http://www.ilovematlab.cn/forum-221-1.html
try                                                       
    index=input('Enter subscript of element to display��');
    disp(['a(',int2str(index),')=',num2str(a(index))])    
catch                                                     
    disp(['Illegal subscript��',int2str(index)])           
    A=lasterr;                                            
    disp(['Type of error��',A])                            
end                                                       
