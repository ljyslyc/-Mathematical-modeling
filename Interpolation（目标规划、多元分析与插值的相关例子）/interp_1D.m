%% һά��ֵ�����ĵ���
% yi=interp1(x��y��xi��'method')
% x��yΪ��ֵ�ڵ�������xiΪ����ֵ��
% method�����ֵ����
%��nearest��  ���ڽ���ֵ
% ��linear��  ���Բ�ֵ��
% ��spline��  ����������ֵ��
% ��cubic��   ������ֵ��
% ȱʡʱ     �ֶ����Բ�ֵ

hours=1:12;
temps=[5 8 9 15 25 29 31 30 22 25 27 24];
h=1:0.1:12;
t=interp1(hours,temps,h,'spline');  
plot(hours,temps,'+',h,t,hours,temps,'r:')    
xlabel('Hour'),ylabel('Degrees Celsius')
