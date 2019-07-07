function V=SolidVolume(fun,x,a,b,type)
%SOLIDVOLUME   ���ö��������������
% V=SOLIDVOLUME(FUN,X,A,B,TYPE)  ����ָ�����͵���ת�������������߷����а���
%                                      Ψһ�����Ա���ʱ���������Ӧ����ת��ͼ��
%
% ���������
%     ---FUN�����ߵĺ�������
%     ---X�������Ա���
%     ---A,B��ָ���Ļ������޺�����
%     ---TYPE����ת�����ͣ�TYPE������ȡֵ��
%               1.'c'��0����֪ƽ���������������
%               2.'x'��1����x����ת����ת��
%               3.'y'��2����y����ת����ת��
% ���������
%     ---V�����ص�ͼ���������ת�����
%
% See also int, GraphicArea

s=symvar(fun);
switch type
    case {0,'c'}
        V=simple(int(fun,x,a,b));
    case {1,'x'}
        V=simple(int(pi*fun^2,x,a,b));
        if length(s)==1
            DrawSolid([0,1,0])
        end
    case {2,'y'}
        V=simple(int(pi*fun^2,x,a,b));
        if length(s)==1
            DrawSolid([1,0,0])
        end
    otherwise
        error('Illegal options.')
end
    function DrawSolid(direction)
        t=linspace(a,b,50);
        [X,Y,Z]=cylinder(subs(fun,x,t),50);
        h1=mesh(X,Y,a+(b-a)*Z);
        hidden off
        hold on
        h2=plot3(t,zeros(size(t)),subs(fun,x,t),'k','LineWidth',2);
        x_Lim=get(gca,'xlim');
        y_Lim=get(gca,'ylim');
        z_Lim=get(gca,'zlim');
        axis([x_Lim,y_Lim,z_Lim])
        h3=arrow([0,0,a],[0,0,b],'Length',20,'BaseAngle',30,...
            'TipAngle',20,'Width',2);
        rotate([h1,h2,h3],direction,90,[0,0,0]);
        if isequal(direction,[0,1,0])
            title('��ת�᣺x��')
            axis([z_Lim,y_Lim,x_Lim])
        elseif isequal(direction,[1,0,0])
            title('��ת�᣺y��')
            axis([x_Lim,z_Lim,y_Lim])
        end
        xlabel('x'); ylabel('y')
        h_legend=legend('��ת��','��ת����');
        set(h_legend,'Position',[0.13 0.87 0.22 0.1]);
    end
end
web -broswer http://www.ilovematlab.cn/forum-221-1.html