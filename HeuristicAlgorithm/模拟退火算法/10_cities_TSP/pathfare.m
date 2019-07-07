function [ objval ] = pathfare( fare , path )
% �� �� · �� path �� �� �� objval
% path Ϊ 1 �� n �� �� �� ���� �� �� �� �� �� �� ˳ �� ��
% fare Ϊ �� �� �� �� �� �� Ϊ �� �� ��
[ m , n ] = size( path ) ;
objval = zeros( 1 , m ) ;
for i = 1 : m
    for j = 2 : n
        objval( i ) = objval( i ) + fare( path( i , j - 1 ) , path( i , j ) ) ;
    end
    objval( i ) = objval( i ) + fare( path( i , n ) , path( i , 1 ) ) ;
end