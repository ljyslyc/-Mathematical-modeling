A=imread('D:\Program Files\MATLAB\R2010a\toolbox\images\imdemos\tire.tif');                                   %��ȡͼ��
B=imnoise(A,'salt & pepper');                         %��ӽ�������
SE = strel('disk',2);
C= imopen(B,SE);                                    %��ͼ����п�������
D= imclose(C,SE);                                   %��ͼ����бպϲ���
figure
subplot(131),imshow(B);                             %��ʾ��ӽ���������ͼ��
subplot(132),imshow(C);                             %��ʾ�������ͼ��
subplot(133),imshow(D);                             %��ʾ�������ͼ��

