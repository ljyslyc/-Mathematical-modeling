% chapter2_3_2.m  ��2.3.2�� ��������

a=uint8(9)	%aΪ�޷���һ���ֽ�����
b=int16(8)	% bΪ�з��������ֽ�����
a/b			% �޷�ֱ������
b=uint8(8)	% b��Ϊ�޷���һ���ֽ�����
a/b			% ��ʱ�������㣬��������ֻ������������

%%
a=realmax('double'),b=realmax('single')		%˫���Ⱥ͵����ȸ����������ֵ
a=realmin('double'),b=realmin('single')		%˫���Ⱥ͵����ȸ���������Сֵ
class(pi)                                   %�������ֵ�Ĭ����������Ϊdouble��
class(2)

%%
ele=1:10	% ����һ������
l=ele>5     % �����д���5��Ԫ��λ��
ele(l)		% ȡ������5��Ԫ��

%%
x=[1,2,3,4]						%����x
ha=@sum							%ֱ������haΪsum�����ľ��
hb=str2func('sum')				%��str2func����hbΪsum�����ľ��
functions(ha)					%�������ha��������Ϣ
functions(hb)					%�������hb��������Ϣ
sum(x)							%ʹ��sum���
ha(x)							%ʹ��ha����sum
hb(x)							%ʹ��hb����sum
feval('sum',x)					%��ʹ�ú��������ʹ��feval�������
hc=@myfun
functions(hc)					%�����������Ϣ
hd=@(x,y)x^(-2)+y^(-2);			%���������������
functions(hd)					%���������������Ϣ
version -java
%%
a={1,2,3}								%1��3ϸ������
b=[{zeros(2,2)},{uint8(9)};{'Matlab'},{0}]			%2��2ϸ������
c=b(3)									%c=b(3)��c��һ��СһЩ��ϸ������
class(c)
d=b{3}									%d=b{3}��dΪuint8������
class(d)
A=cell(2,3)								%��cell���������յ�ϸ������
A{1}=zeros(2,2);
A{2}='abc';
A(3)={uint8(9)};
A

%%
book.name='MATLAB';                                             %ֱ�Ӵ����ṹ������
book.price=20;
book.pubtime='2011';
book
book2=struct('name','Matlab','price',20,'pubtime','2011');		%��struct���������ṹ����
book2
whos

%%
for i=1:10,...						%����10����¼��3���ֶεĽṹ����
        books(i).name=strcat('book',num2str(i));...
        books(i).price=20+i;...
        books(i).pubtime='2011';
end;
books
books(1)
price=[books.price]					%��[]�������ȡ��price�ֶ��γ��µ�����
web -broswer http://www.ilovematlab.cn/forum-222-1.html