close all;										%�رյ�ǰ����ͼ�δ���
clear all;										%��չ����ռ����
clc;											%����
obj = mmreader('xylophone.mpg', 'tag', 'myreader1');		%������ý���ļ��������������ñ�ǩ
Frames = read(obj);								%��ȡ��Ƶ������ÿһ֡ͼ���������Frames��
numFrames = get(obj, 'numberOfFrames');			%��ȡ��Ƶ������֡��
for k = 1 : numFrames						
mov(k).cdata = Frames(:,:,:,k); 				%��ÿһͼ��֡�е����ݾ����ȡ��������mov(k).cdata��
mov(k).colormap = [];						%����ɫ��ֵΪ��
end
hf = figure;								%����һ��ͼ�񴰿�
set(hf, 'position', [150 150 obj.Width obj.Height]); 	%������Ƶ֡�Ŀ�Ⱥ͸߶ȣ���������ͼ�񴰿ڴ�С
movie(hf, mov, 1, obj.FrameRate);				%������Ƶ��ԭ����֡���������Ÿ���Ƶ
