%% ��άԪ���Զ���
%imagesc(a)��ɫ�Ⱦ���a��0->256�������
% �����Ȱ��м����Ϊ1��ÿһʱ��ÿһ�����
%��Χ�˸����Ϊż�������Ϊ0��Ϊ�������Ϊ 1
% ��ɫ����
clc, clear

Map = [1 1 1; 0 0 0];% [0 0 0] is black, [1 1 1] is white,
colormap(Map);
% ���������С
S = 121;
L = zeros(S);
% ���м�һ��������Ϊ 1 ��ΪԪ������
M = (S+1)/2;
display(M)
L(M, M) =1;
Temp = L;
% imagesc(L);

% �������
Layer = (S-1)/2 + 1;
display(Layer)
for t=2:Layer
    for x=M-t+1:M+t-1  % 60-62,59-63...1-121
        if x==M-t+1 || x==M+t-1
            for y=M-t+1:M+t-1
                SUM = 0;
                for m=-1:1
                    for n=-1:1
                        if x+m>0 && x+m<=S && y+n>0 && y+n<=S  % s+121;
                            SUM = SUM + L(x+m, y+n);
                            % display(SUM)
                        end
                    end
                end
                SUM = SUM - L(x, y);
                Temp(x, y) = mod(SUM, 2); % mod(a,b)���������a����b������
            end
        else
            y = M-t+1;
            SUM = 0;
            for m=-1:1
                for n=-1:1
                    if x+m>0 && x+m<=S && y+n>0 && y+n<=S
                        SUM = SUM + L(x+m, y+n);
                    end
                end
            end
            SUM = SUM - L(x, y);
            Temp(x, y) = mod(SUM, 2);
            
            y = M+t-1;
            SUM = 0;
            for m=-1:1
                for n=-1:1
                    if x+m>0 && x+m<=S && y+n>0 && y+n<=S
                        SUM = SUM + L(x+m, y+n);
                    end
                end
            end
            SUM = SUM - L(x, y);
            Temp(x, y) = mod(SUM, 2);
        end
    end
    L = Temp;
    imagesc(L);
    % �ٶȿ���
    pause(0.05);
end
