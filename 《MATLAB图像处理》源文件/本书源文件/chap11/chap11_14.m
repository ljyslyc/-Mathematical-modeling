%����11-14��
I= imread('leaf1.bmp');                     %����ͼ�� ����
I= im2bw(I);                              %ת��Ϊ��ֵͼ��
C=bwlabel(I,4);                           %�Զ�ֵͼ�����4��ͨ�ı��
Ar=regionprops(C,'Area');                  %��C�����
Ce=regionprops(C,'Centroid');              %��C������
Ar
Ce

