close all; clear all; clc;					%�ر�����ͼ�δ��ڣ���������ռ����б��������������
I1=imread('lena.bmp');  					%����ͼ��
I2=I1(:);  								%��ԭʼͼ��д��һά�����ݲ���Ϊ I2
I2length=length(I2); 						%����I2�ĳ���
I3=im2bw(I1,0.5);						%��ԭͼת��Ϊ��ֵͼ����ֵΪ0.5
%���³���Ϊ��ԭͼ������г̱��룬ѹ��
X=I3(:);  								%��XΪ�½��Ķ�ֵͼ���һά������
L=length(X);
j=1;
I4(1)=1;
for z=1:1:(length(X)-1)  					%�г̱�������
if  X(z)==X(z+1)
I4(j)=I4(j)+1;
else
data(j)=X(z);  							% data(j)������Ӧ����������
j=j+1;
I4(j)=1;
end
end
data(j)=X(length(X)); 					%���һ���������ݸ���data
I4length=length(I4);  					%�����г̱�������ռ�ֽ�������ΪI4length
CR=I2length/I4length; 					%�Ƚ�ѹ��ǰ��ѹ����Ĵ�С
%����������г̱����ѹ
l=1;
for m=1:I4length
    for n=1:1:I4(m);
        decode_image1(l)=data(m);
        l=l+1;
    end
end
decode_image=reshape(decode_image1,512,512); %�ؽ���άͼ������ 						
figure,
x=1:1:length(X); 
subplot(131),plot(x,X(x));%��ʾ�г̱���֮ǰ��ͼ������
y=1:1:I4length ;          				
subplot(132),plot(y,I4(y));%��ʾ�����������Ϣ
u=1:1:length(decode_image1);       			
subplot(133),plot(u,decode_image1(u));%�鿴��ѹ���ͼ������
subplot(121);imshow(I3);%��ʾԭͼ�Ķ�ֵͼ��
subplot(122),imshow(decode_image); 			%��ʾ��ѹ�ָ����ͼ��
disp('ѹ����: ')
disp(CR);
disp('ԭͼ�����ݵĳ��ȣ�')
disp(L);
disp('ѹ����ͼ�����ݵĳ��ȣ�')
disp(I4length);
disp('��ѹ��ͼ�����ݵĳ��ȣ�')
disp(length(decode_image1));
