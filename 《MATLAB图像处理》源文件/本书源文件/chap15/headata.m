function [fp,axes_x,axes_y,pixel]=headata(N)
%N��ʾ����ͷ�ļ��Ĵ�С��fp��ʾ�洢ͷģ�������ļ���ָ��ͷ��
%axes_x��axes_y��ʾ���ļ���ͼ���귶Χ��
%pixelΪͷģ�����ݾ���
%���������ʽ[fp,axes_x,axes_y,pixel]=headata(N)��
 lenth=N*N;
 pixel=zeros(N,N);%����ͼ����ܶȾ��󣬳�ֵΪ��
 coordx=[0,0,0.22,-0.22,0,0,0,-0.08,0,0.06];%ÿ����Բ���ĵ�x���꣬������Բ����ͬ��֯
 coordy=[0,-0.0184,0,0,0.35,0.1,-0.1,-0.605,-0.605,-0.605];%ÿ����Բ���ĵ�y���ꣻ
 laxes=[0.92,0.874,0.31,0.41,0.25,0.046,0.046,0.046,0.023,0.046];%ÿ����Բ����Ĵ�С
 saxes=[0.69,0.6624,0.11,0.16,0.21,0.046,0.046,0.023,0.023,0.023];%ÿ����Բ����Ĵ�С
 angle=[90,90,72,108,90,0,0,0,0,90];%ÿ����Բ��ת�ĽǶ�
 density=[2.0,-0.98,-0.4,-0.4,0.2,0.2,0.2,0.2,0.2,0.3];%ÿ����Բ�ĻҶ�ֵ
   for i=1:N,
        for j=1:N,
            for k=1:10,
                axes_x(i,j)=(-1+j*2/N-0.5*2/N);%��ͼ��ʱ��x����
                x=(-1+j*2/N)-coordx(k);
                axes_y(i,j)=(-1+i*2/N-0.5*2/N);%��ͼ��ʱ��y����
                y=(-1+i*2/N)-coordy(k);
                alpha=pi*angle(k)/180;
                a=(x*cos(alpha)+y*sin(alpha))/laxes(k);%�ж����ص��Ƿ��ڵ�k����Բ��
                b=(-x*sin(alpha)+y*cos(alpha))/saxes(k);
                if((a*a+b*b)<=1)
                pixel(i,j)=density(k)+pixel(i,j);
                end
            end
        end
   end
 fp=fopen('datafile_name.txt','w'); %����ͷģ�������ļ����������д����
 for i=1:N,
    for j=1:N,
            a=[i j pixel(i,j)];
            fprintf(fp,'%d  %d  %f\n',a);
    end
 end
fclose(fp);%�ر��ļ�ָ��
fp=fopen('datafile_name.txt','r'); %����datafile_name.txt�ļ�ͷ

 
