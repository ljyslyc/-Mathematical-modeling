clc,clear
syms m V rho g k s(t) v(t) %������ų����ͷ��ű���
ds=diff(s); %����s��һ�׵�����Ϊ�˳�ֵ������ֵ
s=dsolve(m*diff(s,2)-m*g+rho*g*V+k*diff(s),s(0)==0,ds(0)==0);
s=subs(s,{m,V,rho,g,k},{239.46,0.2058,1035.71,9.8,0.6}); %������ֵ
s=simplify(s); %����
s=vpa(s,6)  %��ʾС����ʽ��λ�ƺ���
v=dsolve(m*diff(v)-m*g+rho*g*V+k*v,v(0)==0);
v=subs(v,{m,V,rho,g,k},{239.46,0.2058,1035.71,9.8,0.6});
v=simplify(v); %����
v=vpa(v,6)  %��ʾС����ʽ���ٶȺ���
y=s-90; 
tt=solve(y); tt=double(tt)   %�󵽴ﺣ��90�״���ʱ��
vv=subs(v,tt);vv=double(vv)  %�󵽵׺���90�״����ٶ�
