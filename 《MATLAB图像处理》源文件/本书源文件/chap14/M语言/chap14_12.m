A=imread('D:\Program Files\MATLAB\R2010a\toolbox\images\imdemos\coins.png');                         %��ȡͼ��
B=im2bw(A);                                 %ת���ɶ�ֵͼ��
SE=strel('disk',5);    
C=imopen(B,SE);                             %��ͼ����п�������
figure
subplot(121),imshow(B);                     %��ʾ��ֵͼ��        
subplot(122),imshow(C);                     %��ʾ�������ͼ��