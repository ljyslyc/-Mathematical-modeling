A=imread('D:\Program Files\MATLAB\R2010a\toolbox\images\imdemos\liftingbody.png');			%���벢��ʾԭʼͼ�� 
B=double(A);
B=256-1-B;
B=uint8(B);		                   %ͼ����������ת��                   
figure,imshow(B);               %��ʾ�任��Ľ��
