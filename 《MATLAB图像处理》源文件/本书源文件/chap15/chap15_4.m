clear all;  			%��������ռ䣬�ر�ͼ�δ��ڣ����������
close all;
clc;
N=128;								%����ͼ���С����Բ������ͶӰ�� 
NUM=10;
PROJ=180;
[sp,axes_x,axes_y,pixel]=headata(N) %��������ͷģ������
degree=projdata(PROJ,N);            %��ͷģ�Ͳ���ͶӰ����
[axes_x,h]=RLfilter(N);             %�����˲�����
 m=N;
 n=NUM;
 k=PROJ;
 F=zeros(m,m);                      %������ʼ��
 for k=1:PROJ
    for j=1:N-1                                     
      sn(j)=0;                                     
      for i=1:N-1
        sn(j)=sn(j)+h(j+N-i)*degree(k,i);		%����Qtheta��ͶӰ�������˲��������
      end
    end
    
 for i=1:N
     for j=1:N
       cq=N/2-(N-1)*(cos(k*pi/PROJ)+sin(k*pi/PROJ))/2;			%��ͶӰ�����ؽ�ͼ��
	   s2=((i-1)*cos(k*pi/PROJ)+(j-1)*sin(k*pi/PROJ)+cq);
	   n0=fix(s2);%��������
       s4=s2-n0;%С������
       if((n0>=1) && ((n0+1)<N))
         F(j,i)=F(j,i)+(1.0-s4)*sn(n0)+s4*sn(n0+1);
       end
     end
   end
 end
set(0,'defaultFigurePosition',[100,100,1200,450]);%�޸�ͼ��ͼ��λ�õ�Ĭ������
set(0,'defaultFigureColor',[1 1 1])       %�޸�ͼ�α�����ɫ������ 
figure,
subplot(121),pcolor(pixel);
subplot(122),pcolor(F)					%��ʾ�ع�ͼ��
