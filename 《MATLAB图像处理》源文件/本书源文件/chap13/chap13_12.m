close all;                  %�رյ�ǰ����ͼ�δ��ڣ���չ����ռ��������������ռ����б���
clear all;
clc;
X=imread('6.bmp');                          %��ԭͼ��ת��Ϊ�Ҷ�ͼ��װ�ز���ʾ
X=double(rgb2gray(X)); 
init=2055615866;%���ɺ���ͼ����ʾ
randn('seed',init)
X1=X+25*randn(size(X));%���ɺ���ͼ����ʾ
[thr,sorh,keepapp]=ddencmp('den','wv',X1);%���봦�����ú���wpdencmp���������
X2=wdencmp('gbl',X1,'sym4',2,thr,sorh,keepapp);
X3=X;                                   %���洿����ԭͼ��
for i=2:577;
      for j=2:579
           Xtemp=0;
            for m=1:3
                 for n=1:3
                       Xtemp=Xtemp+X1((i+m)-2,(j+n)-2);%��ͼ�����ƽ����������ǿ����Ч������ֵ�˲���
                end      
            end
            Xtemp=Xtemp/9;
            X3(i-1,j-1)=Xtemp;
      end    
end
set(0,'defaultFigurePosition',[100,100,1000,500]);%�޸�ͼ��ͼ��λ�õ�Ĭ������
set(0,'defaultFigureColor',[1 1 1])       %�޸�ͼ�α�����ɫ������
figure
subplot(121);imshow(uint8(X)); axis square;              %����ԭͼ��
subplot(122);imshow(uint8(X1));axis square;              %����������ͼ��
figure
subplot(121),imshow(uint8(X2)),axis square;%����������ͼ��
subplot(122),imshow(uint8(X3)),axis square;%��ʾ���
Ps=sum(sum((X-mean(mean(X))).^2));%���������
Pn=sum(sum((X1-X).^2));
Pn1=sum(sum((X2-X).^2));
Pn2=sum(sum((X3-X).^2));
disp('δ����ĺ�����ͼ�������')
snr=10*log10(Ps/Pn)
disp('����С��ȫ����ֵ�˲���ȥ��ͼ�������')
snr1=10*log10(Ps/Pn1)
disp('������ֵ�˲���ȥ��ͼ�������')
snr2=10*log10(Ps/Pn2)