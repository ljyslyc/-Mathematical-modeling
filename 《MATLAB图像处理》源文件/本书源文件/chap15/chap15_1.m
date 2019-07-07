close all;                  %�رյ�ǰ����ͼ�δ��ڣ���չ����ռ��������������ռ����б���
clear all;
clc;
N1=256;%����ͷģ�ʹ�С
[fp1,axes_x1,axes_y1,pixel1]=headata(N1);%���ú���headata����ͷģ������

set(0,'defaultFigurePosition',[100,100,1200,450]);%�޸�ͼ��ͼ��λ�õ�Ĭ������
set(0,'defaultFigureColor',[1 1 1])       %�޸�ͼ�α�����ɫ������
figure,                                     %��ʾ256*256ͷģ��            
subplot(121),
 for i=1:N1,
    for j=1:N1,
           a=fscanf(fp1,'%d  %d  %f\n',[1 3]);
           plot(axes_x1(i,j),axes_y1(i,j),'color',[0.5*a(3) 0.5*a(3) 0.5*a(3)],...
                                        'MarkerSize',20);
           hold on;
    end
 end
fclose(fp1);
N2=512;%����ͷģ�ʹ�С
[fp2,axes_x2,axes_y2,pixel2]=headata(N2);%������headata����ͷģ������ 
 subplot(122),                          %��ʾ512*512ͷģ��
 for i=1:N2,
    for j=1:N2,
           a=fscanf(fp2,'%d  %d  %f\n',[1 3]);
           plot(axes_x2(i,j),axes_y2(i,j),'color',[0.5*a(3) 0.5*a(3) 0.5*a(3)],...
                                        'MarkerSize',20);
           hold on;
    end
 end
 fclose(fp2);