%����11-15��
close all;							%�رյ�ǰ����ͼ�δ��ڣ���չ����ռ��������������ռ����б���
clear all;
clc;
I = imread('liftingbody.png');             %����ͼ��
S = qtdecomp(I,.27);                   %�Ĳ����ֽ⣬��ֵΪ0.27
blocks = repmat(uint8(0),size(S));        %��������ΪS�Ĵ�С
for dim = [512 256 128 64 32 16 8 4 2 1];    
  numblocks = length(find(S==dim));    
  if (numblocks > 0)        
    values = repmat(uint8(1),[dim dim numblocks]);%���Ͻ�Ԫ��Ϊ1
    values(2:dim,2:dim,:) = 0;%�����ط�Ԫ��Ϊ0
    blocks = qtsetblk(blocks,S,dim,values);
  end
end
blocks(end,1:end) = 1;  blocks(1:end,end) = 1;
set(0,'defaultFigurePosition',[100,100,1000,500]);	%�޸�ͼ��ͼ��λ�õ�Ĭ������
set(0,'defaultFigureColor',[1 1 1])
figure;subplot(121);imshow(I);
subplot(122), imshow(blocks,[])                   %��ʾ�Ĳ����ֽ��ͼ��
