function [axes_x,h]=RLfilter(N,L)
%��������ֵN
delta=L/N;			%���˲�����������ɢ����λ��
for i=2:2:2*N 	%ż����=0
    h(i)=0; 
end
k=1/delta/delta;         %ԭ����=0
h(N)=k/4;
for i=1:2:N-1
  down=-k/(i*i*pi*pi);   %������=-1/(n^2 ��^2 d^2 )
  h(N+i)=down;
  h(N-i)=down;
end
for i=1:2*N
axes_x(i)=(-1+(i-1)/N);%��ͼ��ʱ��x����
end


