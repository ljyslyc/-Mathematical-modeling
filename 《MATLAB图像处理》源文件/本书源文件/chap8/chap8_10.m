

clear all; close all;
I=imread('peppers.png');
J=rgb2gray(I);
J=J*exp(1);
J(find(J>255))=255;
K=fft2(J);
K=fftshift(K);
L=abs(K/256);
figure;
subplot(121);
imshow(J);
subplot(122);
imshow(uint8(L));

