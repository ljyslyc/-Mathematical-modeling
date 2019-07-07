%[10.10]
close all; clear all; clc;				%�ر�����ͼ�δ��ڣ���������ռ����б��������������
ORIGIN=imread('lena.bmp');			%����ԭʼͼ�� 
%����1��������ɢ���ұ任(FDCT)
fun=@DCT_Measure;
%����2������
B=blkproc(ORIGIN,[8,8],fun);			%�õ��������ϵ��������ԭʼͼ��ߴ���ͬ����Ҫ��һ������
n=length(B)/8; 						%��ÿ��ά�ȷֳɵĿ���
C=zeros(8);						%��ʼ��Ϊ8��8��ȫ0����
for y=0:n-1
    for x=0:n-1
        T1=C(:,[end-7:end]);			%ȡ����һ�����������,T1������8�к����8����ɵ�8*8
        T2=B(1+8*x:8+8*x,1+8*y:8+8*y);
        T2(1)=T2(1)-T1(1);			%ֱ��ϵ�������
        C=[C,T2];					%��C��T2������
    end
end
C=C(:,[9:end]);						%ȥ��C��ǰ8�У�����ǰ���ȫ0
%����4������Code_Huffman( )����ʵ������JPEG�㷨�����еĲ���3��4��5��6��
JPGCode={''};						%�洢�����Ԫ����ʼ��Ϊ�յ��ַ���
for a=0:n^2-1
    T=Code_Huffman(C(:,[1+a*8:8+a*8]));
    JPGCode=strcat(JPGCode,T);
end
sCode=cell2mat(JPGCode);			%��Ԫ��ת��Ϊ����
Fid=fopen('JPGCode.txt','w');			%�ñ���fid���I/O�������ı��ļ�
fprintf(Fid,'%s',sCode);				%��ѹ����sCode���浽�ı��ļ��С���Ӷ����Ǹ���
fclose(Fid);						%�ر�I/O��
[x y]=size(A);
b=x*y*8/length(sCode);
v=8/b; 							%����ѹ���Ⱥ�ѹ��Ч��
disp('JPEGѹ�������ѱ�����JPGCode.txt��!');
disp(['ѹ����Ϊ��',num2str(b),'��ѹ��Ч�ʣ�',num2str(v)]);
