%% �������纯�������Ǻ��Ŵ��㷨Ψһ��ͬ�ĺ�����������������Ⱦɫ��Ľ��������

%chromosomeGroup:Ⱦɫ����
%bachterinChromosome:����Ⱦɫ�壬����õ�Ⱦɫ�塣�����Ⱦɫ����ȡ����
%parameter:��������Ĳ���������ʲô��������
%inoculateChromosome:����������Ⱦɫ��
function inoculateChromosome=immunity(chromosomeGroup,bacterinChromosome,parameter)
[chromosomeGroupSum,chromosomeLength]=size(chromosomeGroup);
[row,bacterinChromosomeLength]=size(bacterinChromosome);
%chromosomeGroupSum:Ⱦɫ���������chromosomeLength��Ⱦɫ��ĳ���
switch parameter
    case 1%���ѡ��Ⱦɫ����н���
        for i=1:chromosomeGroupSum
            %%%%%%%%%%%%������Ⱦɫ���϶�λ����
            headDot=fix(rand(1)*bacterinChromosomeLength);
            %������Ⱦɫ������ߵĵ�λ
            if headDot==0%��ֹ����0��λ
                headDot=1;
            end
            tailDot=fix(rand(1)*bacterinChromosomeLength);
            %������Ⱦɫ�����ұߵĵ�λ
            if tailDot==0%��ֹ����0��λ
                tailDot=1;
            end
            if tailDot>headDot%��ֹ�ұߵĵ�λ������ߵĵ�λ
                dot=headDot;
                headDot=tailDot;
                tailDot=dot;
            end
            %%%%%%%%%%%%%����
            randChromosomeSequence=round(rand(1)*chromosomeGroupSum);
            %�������1��Ⱦɫ�����ţ�������Ⱦɫ����н���
            if randChromosomeSequence==0%��ֹ����0���
                randChromosomeSequence=1;
            end
            inoculateChromosome(i,:)...%�Ȱ�����Ⱦɫ�崫�����
            =chromosomeGroup(randChromosomeSequence,:);
            %ִ�����ߣ���������Ⱦɫ����ȡ��һ�λ��������磬��ע�뵽����Ⱦɫ����
            inoculateChromosome(i,headDot:tailDot)...
            =bacterinChromosome(1,headDot:tailDot);
        end
    case 2 %����Ⱦɫ�尤������
        for i=1:chromosomeGroupSum
        %%%%%%%%%%%%������Ⱦɫ���϶�λ����
            headDot=fix(rand(1)*bacterinChromosomeLength);
            %������Ⱦɫ������ߵĵ�λ
            if headDot==0%��ֹ����0��λ
                headDot=1;
            end
            tailDot=fix(rand(1)*bacterinChromosomeLength);
            %������Ⱦɫ�����ұߵĵ�λ
            if tailDot==0%��ֹ����0��λ
                tailDot=1;
            end
            if tailDot>headDot%��ֹ�ұߵĵ�λ������ߵĵ�λ
                dot=headDot;
                headDot=tailDot;
                tailDot=dot;
            end
            %%%%%%%%%%%%%����
            inoculateChromosome(i,:)=chromosomeGroup(i,:);%�Ȱ�����Ⱦɫ�崫�����
            %ִ�����ߣ���������Ⱦɫ����ȡ��һ�λ��������磬��ע�뵽����Ⱦɫ����
            inoculateChromosome(i,headDot:tailDot)...
            =bacterinChromosome(1,headDot:tailDot);
        end
    case 3 %����λ���������
        for i=1:chromosomeGroupSum
        %%%%%%%%%%%%������Ⱦɫ���϶�λ����
            headDot=fix(rand(1)*bacterinChromosomeLength);
            %������Ⱦɫ������ߵĵ�λ
            if headDot==0%��ֹ����0��λ
                headDot=1;
            end
            tailDot=fix(rand(1)*bacterinChromosomeLength);
            %������Ⱦɫ�����ұߵĵ�λ
            if tailDot==0%��ֹ����0��λ
                tailDot=1;
            end
            if tailDot>headDot%��ֹ�ұߵĵ�λ������ߵĵ�λ
                dot=headDot;
                headDot=tailDot;
                tailDot=dot;
            end
            %%%%%%%%%%%%%��Ⱦɫ���������λ����λ��
            inoculateDot=fix(rand(1)*chromosomeLength);%���ѡ��Ⱦɫ��Ľ��ֵ�λ
            if inoculateDot==0
                inoculateDot=1;
                inoculateChromosome(i,:)=chromosomeGroup(i,:);
                inoculateChromosome(i,inoculateDot:tailDot-headDot+1)...
                =bacterinChromosome(1,headDot:tailDot);
            elseif inoculateDot<=headDot
                inoculateChromosome(i,:)=chromosomeGroup(i,:);
                inoculateChromosome(i,inoculateDot:inoculateDot+tailDot-headDot)...
                =bacterinChromosome(1,headDot:tailDot);
            elseif (chromosomeLength-inoculateDot)>=(tailDot-headDot)
                inoculateChromosome(i,:)=chromosomeGroup(i,:);
                inoculateChromosome(i,inoculateDot:inoculateDot+tailDot-headDot)...
                =bacterinChromosome(1,headDot:tailDot);
            else
                inoculateChromosome(i,:)=chromosomeGroup(i,:);
                inoculateChromosome(i,headDot:tailDot)...
                =bacterinChromosome(1,headDot:tailDot);
            end
        end
end