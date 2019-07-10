function [yc0]= huiseguanlian(x0,r)
global x0
global x1
global r
% ���ݾ�ֵ������
x=x1';          % x��һ��m*nά����ÿһ�д���һ���Ƚϱ�����ԭʼ���ݣ�һ����n���Ƚϱ���
y=x0';          % yʱһ��m*1ά�����ǲο�������ԭʼ����
m_x=x(1,:);                              % m_xΪ�����Ƚϱ���ԭʼ���е�ƽ����
m_y=y(1,:);                              % m_yΪ�ο�����ԭʼ���е�ƽ����  

temp=size(x);                       % size�ó���temp������һ��1*2�������е�һ����Ϊx���������ڶ�����Ϊx������
b=repmat(m_x,[temp(1) 1]);          % ����һ����xͬά���ľ���b���Ҵ˾����������Ϊ��Ӧx������������ƽ����,  
                                    % repmat��A,m,n�������Ĺ����ǽ�����A����m��n�飬��B��m��n��Aƽ�̶���
x_initial=x./b;
y_initial=y./m_y;

% %���ݳ�ֵ������ 
% x0_initial=x0./x0(1); 
% temp=size(x1); 
% b=repmat(x1(:,1),[1 temp(2)]); 
% x1_initial=x1./b; 

% �������ϵ��
K=0.5;                                % �ֱ�ϵ��ѡ��                

y_ext=repmat(y_initial,[1 temp(2)]);  % x_initial��һ��m*1ά����y_ext��x_initial������1*n���õ���m*nά����
contrast_mat=abs(y_ext-x_initial);    % contrast_mat��һ��m*nά���󣬾����е�i��j�б�ʾ��i��ʱ�̵�j���Ƚ�������ο����ж�Ӧ�ڵĲ�ֵ�ľ���ֵ

delta_min=min(min(contrast_mat));     % delta_minΪ��Сֵ�е���Сֵ�������ݳ�ֵ����ʵ��Ϊ�㣨С��ȡС�� 
delta_max=max(max(contrast_mat));     % delta_maxΪ���ֵ�е����ֵ�������ݳ�ֵ����ʵ��Ϊ�㣨����ȡ��
a=delta_min+K*delta_max;              % aΪ����ϵ�����㹫ʽ�ķ��ӣ����ڲ�ͬ�Ƚϱ����������Ǹ���ֵ
b=contrast_mat+K*delta_max;           % bΪ����ϵ�����㹫ʽ�ķ�ĸ
incidence_coefficient=a./b;           % �ɹ���ϵ�����㹫ʽ�õ�����ϵ��incidence_coefficient������һ��m*nά���󣬾����е�i�е�j�б�ʾ��i��ʱ�̵�j���Ƚϱ�����ο������Ĺ����� 

% ���������
r=sum(incidence_coefficient)/temp(1);  % ���������r��r��һ��1*nά����    
