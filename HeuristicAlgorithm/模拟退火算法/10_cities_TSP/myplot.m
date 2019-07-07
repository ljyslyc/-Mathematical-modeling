function [ ] = myplot( path , coord , pathfar )
% �� �� · �� �� ͼ ��
% path Ϊ Ҫ �� ͼ �� · �� ��coord Ϊ �� �� �� �� �� �� ��
% pathfar Ϊ · �� path �� Ӧ �� �� ��
len = length( path ) ;
clf ;
hold on ;
title( [ '�������·�����£�����Ϊ' , num2str( pathfar ) ] ) ;
plot( coord( 1 , : ) , coord( 2 , : ) , 'ok');
pause( 0.4 ) ;
for ii = 2 : len
    plot( coord( 1 , path( [ ii - 1 , ii ] ) ) , coord( 2 , path( [ ii - 1 , ii ] ) ) , '-b');
    x = sum( coord( 1 , path( [ ii - 1 , ii ] ) ) ) / 2 ;
    y = sum( coord( 2 , path( [ ii - 1 , ii ] ) ) ) / 2 ;
    text( x , y , [ '(' , num2str( ii - 1 ) , ')' ] ) ;
    pause( 0.4 ) ;
end
plot( coord( 1 , path( [ 1 , len ] ) ) , coord( 2 , path( [ 1 , len ] ) ) , '-b' ) ;
x = sum( coord( 1 , path( [ 1 , len ] ) ) ) / 2 ;
y = sum( coord( 2 , path( [ 1 , len ] ) ) ) / 2 ;
text( x , y , [ '(' , num2str( len ) , ')' ] ) ;
pause( 0.4 ) ;
hold off ;