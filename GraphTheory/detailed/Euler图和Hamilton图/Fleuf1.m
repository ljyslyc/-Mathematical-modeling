function [ T c ] = Fleuf1(d)
%d��ʾͼ��Ȩֵ����
% T��ʾ�ߵļ��ϣ�c��ʾȨ�غ�
% ע�⣺��֤����ľ�������ʾ��ͼ����ŷ��ͼ��������򽫱���
% ������ŷ����·�������������Ǹû�·�Ķ��㣬���ж�������
% Ȩֵ���󶥵������Ӧ����������ŷ����·���������������Ϣ��T=0,c=0

b = d;
b(b==inf)=0;
b(b~=0)=1;
a = sum(b);
% �ߵĸ���
eds = sum(a)/2;
ed = zeros(2,eds);
vexs = zeros(1,eds+1);
matr = b;

% �ж��Ƿ����ŷ����·
a = mod(a,2);
x = sum(a);
if x ~= 0
    fprintf('There is not exist Euler path\n');
    T = 0;c =0;
end

if x==0
    vet = 1;
    flag = 0;
    t1 = find(matr(vet,:)==1);
    for ii = 1:length(t1)
        ed(:,1)=[vet,t1(ii)];
        vexs(1,1)=vet;vexs(1,2)=t1(ii);
        matr(vexs(1,2),vexs(1,1))=0;
        flagg = 1;tem = 1;
        while flagg
            [flagg ed]=edf(matr,eds,vexs,ed,tem);
            tem = tem + 1;
            if ed(1,eds) ~= 0 && ed(2,eds) ~= 0
                T = ed;
                T(2,eds)=1;
                c = 0;
                for g = 1:eds
                    c = c + d(T(1,g),T(2,g));
                end
                flagg=0;
                break;
            end
        end
    end
end
end

