% hopfield_newhop.m
% ����������
T = [-1,  1;...
     1,  -1]
 
 % ����hopfield����
 net=newhop(T);
 
 % ��ԭƽ��λ�õ�������Ϊ������з���
 Y = sim(net,2,[],T);
 fprintf('����ƽ�����ĵó��Ľ����\n');
 disp(Y);

 % ���µ�ֵ��Ϊ����
 rng(0);
 N=10;
 for i=1:N
     y=rand(1,2)*2-1;
     y(y>0) = 1;
     y(y<0) = -1;
     [Y,a,b]=sim(net,{1,5},[],y');
     if (sum(abs(b))<1.0e-1)
         b=[0,0]';
     end
     fprintf('�� %d ���������: ',i);
     disp(y);
     fprintf('�������:         ');
     disp(b');
end
