function SEASON(month)  
% ���㼸�´���ʲô����
%����SEASON(month)���������monthΪ������
switch month      
case {3,4,5}         
season='spring';   
case {6,7,8}     
season='summer';   
case {9,10,11}       
season='autumn'; 
case{1,2,12}
season='winter';
otherwise        
season= 'Wrong';    
end   
