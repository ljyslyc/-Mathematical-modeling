function Gabor= Gabor_hy(Sx,Sy,f,theta,sigma)
% Sx,Sy���˲������ڳ��ȣ�f������Ƶ�ʣ�theta���˲����ķ���sigma�Ǹ�˹���ķ���
x = -fix(Sx):fix(Sx); %Gabor�˲����Ĵ��ڳ���
y = -fix(Sy):fix(Sy);
[x y]=meshgrid(x,y);
xPrime = x*cos(theta) + y*sin(theta);
yPrime = y*cos(theta) - x*sin(theta);
Gabor = (1/(2*pi*sigma.^2)) .* exp(-.5*(xPrime.^2+yPrime.^2)/sigma.^2).*... %Gabor�˲���
                  (exp(j*f*xPrime)-exp(-(f*sigma)^2/2)); 


