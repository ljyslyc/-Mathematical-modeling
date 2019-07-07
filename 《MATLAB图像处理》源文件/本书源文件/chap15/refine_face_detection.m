function BW=refine_face_detection(I)
% I�Ǵ�ʶ��Ĳ�ɫͼ��BW�Ǽ�⵽��ֵ����ͼ��
%%��ɫ����
I1=I;             %����ͼ�����I
R=I1(:,:,1);      %��ȡRGBͼ�����I��R��G��Bȡֵ
G=I1(:,:,2);
B=I1(:,:,3);
Y=0.299*R+0.587*G+0.114*B; %������ɫ�ռ�ת�� ����Y ��Cb
Cb=-0.1687*R-0.3313*G+0.5000*B+128;
for Cb=133:165
    r=(Cb-128)*1.402+Y;  %��YCrCb�ռ���Cb=133:165�е�����ȷ��
    r1=find(R==r);       %������ɫ����Ķ�ֵ����
    R(r1)=255;           %�Է�ɫ���������
    G(r1)=255;
    B(r1)=255;
end
I1(:,:,1)=R;            %���ɷ�ɫ������ͼ��
I1(:,:,2)=G;
I1(:,:,3)=B;
J=im2bw(I1,0.99);       %ת���ɻҶ�ͼ��
%% ���ͺ͸�ʴ
SE1=strel('square',8);
BW1=imdilate(J,SE1);%��С�������
BW1=imfill(BW1,'holes');%���������Ķ�
SE1=strel('square',20);
BW1=imerode(BW1,SE1);%������ĸ�ʴ
SE1=strel('square',12);
BW1=imdilate(BW1,SE1);%���ͣ��ָ���������
%% ��λ�����Ĵ�������
[B,L,N]=bwboundaries(BW1,'noholes');%�߽����
a=zeros(1,N);
for i1=1:N
    a(i1)=length(find(L==i1)); %��ȡ�ߵ�λ��
end
a1=find(a==max(a));
L1=(abs(L-a1))*255;  
I2=double(rgb2gray(I)); %ԭ��ɫͼ��ת�Ҷ�ͼ��
I3=uint8(I2-L1);        %�����ߵ�
BW=I3;                   %���ؽ��