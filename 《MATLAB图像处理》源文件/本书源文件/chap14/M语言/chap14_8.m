A=imread('D:\Program Files\MATLAB\R2010a\toolbox\images\imdemos\kids.tif');   %��ȡ����ʾͼ��
B=imresize(A,0.5,'nearest');  %��Сͼ����ԭʼͼ���50%
figure(1)
imshow(A);                    %��ʾԭʼͼ��   
figure(2)
imshow(B);                    %��ʾ��С���ͼ��
