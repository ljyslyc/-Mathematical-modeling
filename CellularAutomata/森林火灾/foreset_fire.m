%% ɭ�ֻ���
% (1)����ȼ�յ�����ɿո�λ��
% (2)���������λ������ھ�����һ������ȼ�գ�
%   �����������ȼ�յ�����
% (3)�ڿո�λ�����Ը���p������
% (4)��������ھ���û������ȼ�յ������������
%   ��ÿһʱ���Ը���f(����)��Ϊ����ȼ�յ�����
clear
clc
clear all
%��ͼ��С
n=100;
%��������еĸ���
Plightning =0.000005;
%�յ����������ĸ���
Pgrowth = 0.01;
z=zeros(n,n);
o=ones(n,n);
veg=z;
sum=z;
imh = image(cat(3,z,veg*0.02,z));
set(imh, 'erasemode', 'none')
axis equal
axis tight
for i=1:3000
    sum = (veg(1:n,[n 1:n-1])==1) + (veg(1:n,[2:n 1])==1) + ...
        (veg([n 1:n-1], 1:n)==1) + (veg([2:n 1],1:n)==1) ;
    veg = ...
        2*(veg==2) - ((veg==2) & (sum>0 | (rand(n,n)<Plightning))) + ...
        2*((veg==0) & rand(n,n)<Pgrowth) ;    
    set(imh, 'cdata', cat(3,(veg==1),(veg==2),z) )
    pause(0.02)
end
