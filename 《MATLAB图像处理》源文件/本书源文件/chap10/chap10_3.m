%����10-3��
close all; clear all; clc;				%�ر�����ͼ�δ��ڣ���������ռ����б��������������
I=imread('lena.bmp');
I=im2double(I)*255;
[height,width]=size(I);				%��ͼ��Ĵ�С
HWmatrix=zeros(height,width);
Mat=zeros(height,width);				%������С��ԭͼ���С��ͬ�ľ���HWmatrix��Mat������Ԫ��Ϊ0��
HWmatrix(1,1)=I(1,1);				%ͼ���һ������ֵI(1,1)����HWmatrix(1,1)
for i=2:height						%���½�ͼ������ֵ���ݸ�����Mat
    Mat(i,1)=I(i-1,1);
end
for j=2:width
    Mat(1,j)=I(1,j-1);
end
for i=2:height						%���½��������������symbols��ÿ�����س��ֵĸ��ʾ���p
    for j=2:width
        Mat(i,j)=I(i,j-1)/2+I(i-1,j)/2;
    end
end
Mat=floor(Mat);HWmatrix=I-Mat;
SymPro=zeros(2,1); SymNum=1; SymPro(1,1)=HWmatrix(1,1); SymExist=0;
for i=1:height
    for j=1:width
        SymExist=0;
        for k=1:SymNum
            if SymPro(1,k)==HWmatrix(i,j)
                SymPro(2,k)=SymPro(2,k)+1;
                SymExist=1;
                break;
            end
        end
        if SymExist==0
          SymNum=SymNum+1;
          SymPro(1,SymNum)=HWmatrix(i,j);
          SymPro(2,SymNum)=1;
        end
    end
end
for i=1:SymNum
    SymPro(3,i)=SymPro(2,i)/(height*width);
end
symbols=SymPro(1,:);p=SymPro(3,:);
[dict,avglen] = huffmandict(symbols,p);			%��������������ʵ䣬���ر���ʵ�dict��ƽ���볤avglen
actualsig=reshape(HWmatrix',1,[]);
compress=huffmanenco(actualsig,dict); 			%����dict��actuals�����룬���������compress��
UnitNum=ceil(size(compress,2)/8); 
Compressed=zeros(1,UnitNum,'uint8');
for i=1:UnitNum
    for j=1:8
        if ((i-1)*8+j)<=size(compress,2)
        Compressed(i)=bitset(Compressed(i),j,compress((i-1)*8+j));
        end
    end
end
NewHeight=ceil(UnitNum/512);Compressed(width*NewHeight)=0;
ReshapeCompressed=reshape(Compressed,NewHeight,width);
imwrite(ReshapeCompressed,'Compressed Image.bmp','bmp');
Restore=zeros(1,size(compress,2));
for i=1:UnitNum
    for j=1:8
        if ((i-1)*8+j)<=size(compress,2)
        Restore((i-1)*8+j)=bitget(Compressed(i),j);
        end
    end
end
decompress=huffmandeco(Restore,dict); 			%����dict��Restore�����룬���������decompress��
RestoredImage=reshape(decompress,512,512);
RestoredImageGrayScale=uint8(RestoredImage'+Mat);
imwrite(RestoredImageGrayScale,'Restored Image.bmp','bmp');
figure;
subplot(1,3,1);imshow(I,[0,255]);				%��ʾԭͼ
subplot(1,3,2);imshow(ReshapeCompressed);		%��ʾѹ�����ͼ��
subplot(1,3,3);imshow('Restored Image.bmp');		%��ѹ���ͼ��

