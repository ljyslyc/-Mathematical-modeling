clc, clear
x=-3:0.1:3;y=-5:0.1:5;
[x,y]=meshgrid(x,y); %������������
z=(sin(x.*y)+eps)./(x.*y+eps); %Ϊ����0/0,���ӷ�ĸ����eps
mesh(x,y,z)

