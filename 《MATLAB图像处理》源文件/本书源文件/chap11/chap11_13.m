%����11-13��
I= imread('leaf1.bmp');                           %����ͼ�� ����
c= im2bw(I, graythresh(I));                        %Iת��Ϊ��ֵͼ��
set(0,'defaultFigurePosition',[100,100,1000,500]);	 %�޸�ͼ��ͼ��λ�õ�Ĭ������
set(0,'defaultFigureColor',[1 1 1])
figure;subplot(131);imshow(I);                     %��ʾԭͼ
c=flipud(c);                                      %ʵ�־���c���·�ת              
b=edge(c,'canny');                               %����canny���ӽ���������ȡ
[u,v]=find(b);                                    %���ر߽����b�з���Ԫ�ص�λ��
xp=v;                                          %��ֵv����xp
yp=u;                                          %��ֵu����yp 
x0=mean([min(xp),max(xp)]);                     %x0Ϊ��ֵ�ľ�ֵ
y0=mean([min(yp),max(yp)]);                      %y0Ϊ��ֵ�ľ�ֵ
xp1=xp-x0;
yp1=yp-y0;
[cita,r]=cart2pol(xp1,yp1);                         %ֱ������ת���ɼ�����
q=sortrows([cita,r]);                              %��r�п�ʼ�Ƚ���ֵ������������
cita=q(:,1);                                      %���Ƕ�ֵ
r=q(:,2);                                         %���뾶ģֵ
subplot(132);polar(cita,r);                          %�����������µ�����ͼ
[x,y]=pol2cart(cita,r);
x=x+x0;
y=y+y0;
subplot(133);plot(x,y);                            %����ֱ�������µ�����ͼ
