function N1 = Riemann_closeness(x)
    % �������������
    N1 = quadl(@minimun, 0, 100) / quadl(@maximun, 0, 100);
    
end