I=imread('D:\Program Files\MATLAB\R2010a\toolbox\images\imdemos\pout.tif ');%���벢��ʾԭʼͼ��
I=double(I);                        %ͼ����������ת��
[M,N]=size(I);
for i=1:M				            %�������лҶȱ任
for j=1:N
	    if I(i,j)<=30	 	
I(i,j)=I(i,j);
         elseif I(i,j)<=150
           I(i,j)=(200-30)/(150-30)*(I(i,j)-30)+30;  
         else
           I(i,j)=(255-200)/(255-150)*(I(i,j)-150)+200;
        end
end
end
figure,imshow(uint8(I));           %��ʾ�任��Ľ��
