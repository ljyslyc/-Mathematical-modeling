A=imread('D:\Program Files\MATLAB\R2010a\toolbox\images\imdemos\fuwa.jpg ');                %���벢��ʾͼ��
B=fspecial('Sobel');                        %��Sobel���ӽ��б�Ե��
fspecial('Sobel');
B=B';                                       %Sobel��ֱģ��
C=filter2(B,A);
figure,imshow(C);                        %��ʾͼ��
