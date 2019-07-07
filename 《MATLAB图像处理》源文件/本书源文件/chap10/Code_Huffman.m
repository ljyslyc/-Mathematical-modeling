function B=Code_Huffman(A)
%����huffman����������������ݱ���
%��������DCϵ����ֵ(A��DCϵ�����������)��ACϵ���ĵ���Huffman��
%ֻ����8��8DCTϵ����������ÿ��A����8*8
DC_Huff={'00','010','011','100','101','110','1110','11110','111110','1111110','11111110','111111110'};
%����ACϵ���������ϴ����ǽ���������AC_Huff.txt�ļ��У���������Ԫ��������
fid=fopen('AC_Huff.txt','r');
AC_Huff=cell(16,10);
for a=1:16
    for b=1:10
        temp=fscanf(fid,'%s',1);%����Ϊ��λ��ȡ��������temp��
        AC_Huff(a,b)={temp};%����ÿ�е�һ������
    end
end
fclose(fid);
%��A�е����ݽ���Zig-Zagɨ�裬����������Z��
i=1;
for a=1:15
    if a<=8
        for b=1:a
            if mod(a,2)==1
               Z(i)=A(b,a+1-b);
               i=i+1;
            else
               Z(i)=A(a+1-b,b);
               i=i+1;
            end
            
        end
    else
        for b=1:16-a
            if mod(a,2)==0
               Z(i)=A(9-b,a+b-8);
               i=i+1;
            else
               Z(i)=A(a+b-8,9-b);
               i=i+1;
            end                              
        end
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%�ȶ�DC��ֵϵ�����룺ǰ׺��SSSS+β��
%dcΪ��Huffman����
if Z(1)==0
    sa.s=DC_Huff(1);   %%%size�������ǰ׺��
    sa.a='0';           %%%amp�������β��
    dc=strcat(sa.s,sa.a);
else    
    n=fix(log2(abs(Z(1))))+1;
    sa.s=DC_Huff(n);
    sa.a=binCode(Z(1));
    dc=strcat(sa.s,sa.a);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%�ٶ�ACϵ�������г̱���,�����ڽṹ������rsa��
if isempty(find(Z(2:end)))  %���63������ϵ��ȫ��Ϊ0��rsaϵ��ȫ��Ϊ0
    rsa(1).r=0;             %�г�runlength
    rsa(1).s=0;             %�볤size
    rsa(1).a=0;             %�����Ʊ���
else
   T=find(Z);              %�ҳ�Z�з���Ԫ�ص��±�
   T=[0 T(2:end)];         %Ϊͳһ������һ���±�Ԫ����Ϊ0
   i=1;                    % iΪrsa�ṹ����±�  
   %�ӵڶ���Ԫ�ؼ���һ������Ԫ�ؿ�ʼ����
   j=2;
   while j<=length(T)
       t=fix((T(j)-1-T(j-1))/16);   %�ж��±����Ƿ񳬹�16
       if t==0                      %���С��16���ϼ�
           rsa(i).r=T(j)-T(j-1)-1;
           rsa(i).s=fix(log2(abs(Z(T(j)))))+1;
           rsa(i).a=Z(T(j));
           i=i+1;
       else                         %�������16����Ҫ����15��0�����������
           for n=1:t                %���ܳ���t�飨15��0�� 
               rsa(i)=struct('r',15,'s',0,'a',0);
               i=i+1;
           end
           %���Ŵ���ʣ����ǲ���
           rsa(i).r=T(j)-1-16*t;
           rsa(i).s=fix(log2(abs(Z(T(j)))))+1;
           rsa(i).a=Z(T(j));
           i=i+1;
       end
       j=j+1;
   end
   %�ж����һ������Ԫ���Ƿ�ΪZ�����һ��Ԫ��
   if T(end)<64
       rsa(i).r=0;
       rsa(i).s=0;
       rsa(i).a=0;
   end                      %��EOB����
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%---------ͨ������ȡACϵ����Huffman����

B=dc;                         %B��ʼ��Ϊֱ��ϵ������
for n=1:length(rsa)
    if rsa(n).r==0&rsa(n).s==0&rsa(n).a==0
        ac(n)={'1010'};  
    elseif rsa(n).r==15&rsa(n).s==0&rsa(n).a==0
        ac(n)={'11111111001'};
    else
        t1=AC_Huff(rsa(n).s+1,rsa(n).s);
        t2=binCode(rsa(n).a);
        ac(n)=strcat(t1,t2);
      
    end
   B=strcat(B,ac(n));
end    
    
    
