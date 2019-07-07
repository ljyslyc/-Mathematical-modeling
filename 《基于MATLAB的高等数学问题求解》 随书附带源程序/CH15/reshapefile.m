function varargout=reshapefile(file,n,extension,type)
%RESHAPEFILE   Ϊ����ĸ��������Ż�ɾ��������е�ǰn���ַ�
% RESHAPEFILE(FILE,N)  ΪFILE�ļ��ĸ��������Ų�����ΪTXT�ļ�
% RESHAPEFILE(FILE,N,EXT)�� RESHAPEFILE(FILE,N,EXT,1)��
% RESHAPEFILE(FILE,N,EXT,'add')  ΪFILE�ļ��ĸ��������Ų�
%                                ����Ϊ��չ����EXTָ�����ļ�
% RESHAPEFILE(FILE,N,EXT,2)��
% RESHAPEFILE(FILE,N,EXT,'delete')  ɾ��FILE�ļ����е�ǰN���ַ���
%                                   ����Ϊ��չ����EXTָ�����ļ�
% S=RESHAPEFILE(...)  �����κ���ļ����ݸ���S
%
% ���������
%     ---FILE��ԭ�ļ��������������չ��
%     ---N��ָ�������ռ��λ������Ҫɾ���ַ��ĸ���
%     ---EXT�����ļ�����չ��
%     ---TYPE��ָ����Ϊ�ļ����ݸ��������Ż���ɾ���ļ�����ǰ����ַ���
%              TYPE����������ȡֵ��
%              1.1��'add'��Ϊ�ļ�����������
%              2.2��'delete'��ɾ���ļ�����ǰ��N���ַ�
% ���������
%     ---S�����ļ�������
%
% See also importdata, fprintf

if nargin<4
    type='add';
end
if nargin<3
    extension='.txt';
end
if ~(isnumeric(n) && isscalar(n))
    error('n must be a scalar numeric.')
end
fid=fopen(file);
tline = fgetl(fid);
count=1;
while ischar(tline)
    S{count}=tline;
    tline = fgetl(fid);
    count=count+1;
end
fclose(fid);
N=length(S);
T=cell(1,N);
for k=1:N
    n=ceil(abs(n));
    L=S{k};
    if any(strcmpi(num2str(type),{'1','add'}))
        L=[sprintf(['%0',num2str(n),'d.',blanks(1)],k),L];
    elseif any(strcmpi(num2str(type),{'2','delete'}))
        if n>length(L)
            n=length(L);
        end
        L(1:n)=[];
    end
    T{k}=L;
end
str=char(T);
if nargout==0
    [~,name]=fileparts(file);
    fid=fopen([name,extension],'wt');
    format=[repmat('%c',1,size(str,2)),'\n'];
    fprintf(fid,format,str');
    fclose(fid);
elseif nargout==1
    varargout{1}=str;
else
    error('Illegal number of output arguments.')
end
web -broswer http://www.ilovematlab.cn/forum-221-1.html