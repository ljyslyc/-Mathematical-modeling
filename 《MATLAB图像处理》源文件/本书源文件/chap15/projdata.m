function degree=projdata(proj,N)
%proj��ʾͶӰ��;
%N��ʾͶӰ��R�Ĳ�������;
%����ĵ��ø�ʽdegree=projdata(proj,N)
NUM=10;%��Բ�ĸ���
%ellipse����ʮ����Բ������һ��[]����һ����Բ��ز�����ÿ����Բ��Ӧ��ͬ��֯
%[]��Բ�Ķ�������X0��Y0�����ᣬ���ᣬ��ת�Ƕȣ��Ҷ�ֵ
%����һ����Բ���󣬽�ʮ����Բ������������õ���cell��ʽ
ellipse={[0,0,0.92,0.69,90,2.0],    
         [0,-0.0184,0.874,0.6624,90,0.98],
         [0.22,0,0.31,0.11,72,-0.4],
         [-0.22,0,0.41,0.16,108,-0.4],
         [0,0.35,0.25,0.21,90,0.4],
         [0,0.1,0.046,0.046,0,0.4],
         [0,-0.1,0.046,0.046,0,0.4],
         [-0.08,-0.605,0.046,0.023,0,0.4],
         [0,-0.605,0.023,0.023,0,0.4],
         [0.06,-0.605,0.046,0.023,90,0.4]};
  step=180/proj;%ͶӰ����ת������
  for i=1:NUM
   a(i)=ellipse{i,1}(3);%��i����Բ�ĳ���
   b(i)=ellipse{i,1}(4);%��i����Բ�Ķ���
   c(i)=2*a(i)*b(i);% 2*����*����
   a2(i)=a(i)*a(i);%����ƽ��������a2 1*10
   b2(i)=b(i)*b(i);%����ƽ��������b2 1*10
   alpha(i)=ellipse{i,1}(5)*pi/180;%��i����Բ��ת�ĽǶ�ת���ɻ���
   sina(i)=sin(alpha(i));%sin(alpha)
   cosa(i)=cos(alpha(i));%cos(alpha)
  end
 for j=1:proj
   for i=1:NUM
     theta(j)=step*j*pi/180;%thetaͶӰ�ߵ���x��н�
     angle(i,j)=alpha(i)-theta(j);%alpha��ʾ��Բ����������x��н�;
     zx2(i,j)=sin(angle(i,j))*sin(angle(i,j));%zx2=sinƽ��
     yx2(i,j)=cos(angle(i,j))*cos(angle(i,j));%yx2=cosƽ��
   end
 end
 length=2/N;
for i=1:proj
  R=-(N/2)*length;
  for j=1:N
    R=R+length;
    degree(i,j)=0;
      for m=1:10
        A=a2(m)*yx2(m,i)+b2(m)*zx2(m,i);%a2(m)��Ӧ��Բ����a��ƽ����yx2(m,i)����ƽ��
        x0=ellipse{m,1}(1);
        y0=ellipse{m,1}(2);
        B=R-x0*cos(theta(i))-y0*sin(theta(i));%����ͶӰֵ
        B=B*B;
        E=A-B;
        if (E>0) 
         midu=ellipse{m,1}(6)*c(m)*sqrt(E)/A;
         degree(i,j)=degree(i,j)+midu;
        end
      end 
  end
end





