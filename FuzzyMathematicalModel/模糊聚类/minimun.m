function f1 = minimun(x)
% ģ����A��B�ϵļ�С����
    f1 = (x>=20 & x<50).*(x-20)/40 + (x>=50 & x<80).*(80-x)/40;  
end