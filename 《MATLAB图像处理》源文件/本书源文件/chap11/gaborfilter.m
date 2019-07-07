%garborfilter()���壬IΪ����ͼ��Sx��Sy�Ǳ�����x��y��仯�ķ�Χ����ѡ����gaborС�����ڵĴ�С��fΪ
%���Һ�����Ƶ�ʣ�thetaΪgabor�˲����ķ���GΪgabor�˲�����g(x,y)��gaboutΪgabor�˲����ͼ��
%��άgabor�˲�����:
%                            -1     xp ^     yp  ^             
%%% G(x,y,theta,f) =  exp ([----{(----) 2+(----) 2}])*cos(2*pi*f*xp);
%                             2    Sx      Sy
%%% xp = x*cos(theta)+y*sin(theta);
%%% yp = y*cos(theta)-x*sin(theta);

function [G,gabout] = gaborfilter(I,Sx,Sy,f,theta);     
if isa(I,'double')~=1                                      %�ж�����ͼ�������Ƿ�Ϊdouble���͡�
    I = double(I);                                        %�����ǽ�I��Ϊdouble����
end
for x = -fix(Sx):fix(Sx)                                    %ѡ�����ڴ�С
    for y = -fix(Sy):fix(Sy)                                %��G
        xp = x * cos(theta) + y * sin(theta);
        yp = y * cos(theta) - x * sin(theta);
        G(fix(Sx)+x+1,fix(Sy)+y+1) = exp(-.5*((xp/Sx)^2+(yp/Sy)^2))*cos(2*pi*f*xp);
    end
end
Imgabout = conv2(I,double(imag(G)),'same');               %��ͼ���鲿����ά���
Regabout = conv2(I,double(real(G)),'same');                %��ͼ������ʵ������ά���
gabout = sqrt(Imgabout.*Imgabout + Regabout.*Regabout);  %gaborС���任���ͼ��gabout 
