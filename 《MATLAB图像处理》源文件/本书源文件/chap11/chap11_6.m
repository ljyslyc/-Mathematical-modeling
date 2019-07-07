%����11-6��
%����1����������غ���zxcor()������zxcor.m�ļ�
function [epsilon,eta,C]=zxcor(f,D,m,n)  
%����غ���zxcor(),fΪ�����ͼ�����ݣ�DΪƫ�ƾ��룬��m��n����ͼ��ĳߴ����ݣ�����ͼ����غ���C��
%ֵ��epsilon��eta������غ���C��ƫ�Ʊ�����
for epsilon=1:D									%ѭ�����ͼ��f(x,y)��ƫ��ֵΪD������֮������ֵ
  for eta=1:D                
     temp=0;
     fp=0;
     for x=1:m
        for y=1:n
           if(x+ epsilon -1)>m|(y+ eta -1)>n
             f1=0;
           else   
            f1=f(x,y)*f(x+ epsilon -1,y+ eta -1);     
           end
           temp=f1+temp;
           fp=f(x,y)*f(x,y)+fp;
        end      
     end 
        f2(epsilon, eta)=temp;
        f3(epsilon, eta)=fp;
        C(epsilon, eta)= f2(epsilon, eta)/ f3(epsilon, eta);		%���ֵC
   end
end
epsilon =0:(D-1);									% �����ȡֵ��Χ
eta =0:(D-1);										% �����ȡֵ��Χ
%����2������zxcor()������������ͬͼ�������������
f11=imread('wall.jpg');								%����שǽ��ͼ��ͼ�����ݸ�ֵ��f
f1=rgb2gray(f11);									%��ɫͼ��ת���ɻҶ�ͼ��
f1=double(f1);										%ͼ�����ݱ�Ϊdouble����
[m,n]=size(f1);										%ͼ���С��ֵΪ[m,n]
D=20;											%ƫ����Ϊ20
[epsilon1,eta1,C1]=zxcor1(f1,D,m,n);						%��������غ���
f22=imread('stone.jpg');								%�������ʯͼ��ͼ�����ݸ�ֵ��f
f2=rgb2gray(f22);
f2=double(f2);
[m,n]=size(f2);
[epsilon2,eta2,C2]=zxcor1(f2,20,m,n);					%��������غ���
set(0,'defaultFigurePosition',[100,100,1000,500]);			%�޸�ͼ��ͼ��λ�õ�Ĭ������
set(0,'defaultFigureColor',[1 1 1]);
figure;subplot(121);imshow(f11);
subplot(122);imshow(f22);
figure;subplot(121);mesh(epsilon1,eta1,C1);				%��ʾ����غ�����x��y����άͼ��
xlabel(' epsilon ');ylabel(' eta ');							%��ʾ���������
subplot(122);mesh(epsilon2,eta2,C2);	
xlabel(' epsilon ');ylabel(' eta ');	

