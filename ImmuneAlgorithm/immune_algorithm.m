%% �����㷨
%����㷨�������Ŵ��㷨һ����ֻ�Ƕ�����һ�����ߺ���
%�����㷨���Ŵ��㷨�ı��壬�������ӽ������ǲ���ע������ķ�����
%����������Ⱦɫ���е�һ�λ��򣬰�������ֵ�����Ⱦɫ����

%ע�⣺��׼�Ŵ��㷨��һ����Ҫ�����ǣ�Ⱦɫ���ǿ��ܽ��2����˳��ţ����������ڿ��ܽ�ļ���(��ռ�)���ҵ����ܽ�
%���������㷨������������Ҫ���õĺ������¡�
%�������纯����
%function inoculateChromosome=immunity(chromosomeGroup,bacterinChromosome,parameter)
%parameter:1,�����ȡȾɫ����֡�2��ÿ��Ⱦɫ�嶼���֡�3��ÿ��Ⱦɫ�嶼���֣������ֵ�λ���������
%�������ʵ�ֶ�Ⱦɫ����������

%��Ⱦɫ��(���ܽ��2����)˳����ҵ����ܽ⣺
%x=chromosome_x(fatherChromosomeGroup,oneDimensionSet,solutionSum);

%�ѽ��������Է��������������functionError=nonLinearSumError1(x);

%�ж����Ƿ�ý⺯����[solution,isTrue]=isSolution(x,funtionError,solutionSumError);

%ѡ������Ⱦɫ�庯����
%[bestChromosome,leastFunctionError]=best_worstChromosome(fatherChromosomeGroup,functionError);

%���ȽϺ�����������Ⱦɫ���У�ѡ������С��Ⱦɫ��
%[holdBestChromosome,holdLeastFunctionError]...
% =compareBestChromosome(holdBestChromosome,holdLeastFunctionError,...
% bestChromosome,leastFuntionError)
%ΪȾɫ�嶨����ʺ������õ�Ⱦɫ����ʸߣ���Ⱦɫ����ʵ�
%p=chromosomeProbability(functionError);

%������ѡ��Ⱦɫ�庯����
%slecteChromosomeGroup=selecteChromome(fatherChromosomeGroup,p);

%����Ⱦɫ���ӽ������Ӵ�Ⱦɫ�庯��
%sonChrmosomeGroup=crossChromosome(slecteChromosomeGroup,2);

%��ֹȾɫ�峬����ռ�ĺ���
%chromosomeGroup=checkSequence(chromosomeGroup,solutionSum)

%���캯��
%fatherChromosomeGroup=varianceCh(sonChromosomeGroup,0.8,solutionN);

%ͨ��ʵ�������½����
%1��Ⱦɫ��Ӧ����һЩ
%2��ͨ������ѡ��Ⱦɫ�壬�ڵ������ڻ���Чѡ�������Ⱦɫ�壬ʹ������Ѹ�ٽ��ͣ�
%�����ŵ����Ľ��У�����ѡ��Ҳ�ᵼ��ĳ��Ⱦɫ���ڻ������Ѹ�����ӣ�ʹȾɫ����ͬ��
%��ͼ��������ֵĶ����ԣ��������Աƽ���
%3�����ø���ѡ�񣬽�����Ⱦɫ���ӽ������ñ�������Ⱦɫ�壬Ҳ���Եõ���
%4����������Ч�����ã��ӽ�+����Ч���ȽϺ�

%%%%%%%%%%%%%%%%%%%%%%%%����ʼ����

clear,clc;%�����ڴ棬����
circleN=200;%��������
format long

%%%%%%%%%%%%%%%������ܽ�Ŀռ䣬ȷ��Ⱦɫ��ĸ���������
solutionSum=4;leftBoundary=-10;rightBoundary=10;
distance=1;chromosomeSum=500;solutionSumError=0.1;
%solutionSum:�����Է������Ԫ��(��������ĸ���)��leftBoundary:���ܽ����߽磻
%rightBoundary:���ܽ���ұ߽磻distance:���ܽ�ļ����Ҳ�ǽ�ľ���
%chromosomeSum:Ⱦɫ��ĸ�����solveSumError:������
oneDimensionSet=leftBoundary:distance:rightBoundary;
%oneDimensionSet:���ܽ���һ������(ά)�ϵļ���
oneDimensionSetN=size(oneDimensionSet,2);%����oneDimensionSet�е�Ԫ�ظ���
solutionN=oneDimensionSetN^solutionSum;%��ռ�(�⼯��)�п��ܽ������
binSolutionN=dec2bin(solutionN);%�ѿ��ܽ������ת���ɶ�������
chromosomeLength=size(binSolutionN,2);%�ɽ�ռ��п��ܽ������(��������)����Ⱦɫ��ĳ���

%%%%%%%%%%%%%%%%�����ʼ��
%������ɳ�ʼ���ܽ��˳���,+1��Ϊ�˷�ֹ����0˳���
solutionSequence=fix(rand(chromosomeSum,1)*solutionN)+1;
for i=1:chromosomeSum%��ֹ���˳��ų�����ĸ���
if solutionSequence(i)>solutionN;
solutionSequence(i)=solutionN;
end
end
%Ⱦɫ���ǽ⼯���е����,����Ӧһ�����ܽ�
%�ѽ��ʮ�������ת�ɶ��������
fatherChromosomeGroup=dec2bin(solutionSequence,chromosomeLength);
holdLeastFunctionError=Inf;%���ܽ����С���ĳ�ֵ
holdBestChromosome=0;%��Ӧ��С����Ⱦɫ��ĳ�ֵ

%%%%%%%%%%%%%%%%%%��ʼ����
compute=1;
circle=0;
while compute%��ʼ�������
%%%%%%%%%%%%%1:�ɿ��ܽ�����Ѱ�ҽⱾ��(�ؼ�����)
x=chromosome_x(fatherChromosomeGroup,oneDimensionSet,solutionSum);
%%%%%%%%%%%%%2���ѽ��������Է��̼������
functionError=nonLinearSumError1(x);%�ѽ���뷽�̼������
[solution,minError,isTrue]=isSolution(x,functionError,solutionSumError);
%isSolution�����������functionError�ж������Ƿ��Ѿ��⿪��isTrue=1,���̵ý⡣solution�Ƿ��̵Ľ�
    if isTrue==1
        '���̵ý�'
        solution
        minError
        return%��������
    end
    %%%%%%%%%%%%%3��ѡ����ý��Ӧ������Ⱦɫ��
    [bestChromosome,leastFunctionError]=best_worstChromosome(fatherChromosomeGroup,functionError);
    %%%%%%%%%%%%%4������ÿ�ε�����������õ�Ⱦɫ��
    %������ý����ϴ���ý���бȽϣ�����ϴ���ý����ڱ�����ý⣬�����ϴ���ý⣻
    %��֮������������ý⡣���������Ⱦɫ�����holdBestChromosome��
    [holdBestChromosome,holdLeastFunctionError]...
    =compareBestChromosome(holdBestChromosome,holdLeastFunctionError,...
    bestChromosome,leastFunctionError);
    circle=circle+1
    %minError
    %solution
    holdLeastFunctionError
    if circle>circleN
        return
    end
    %%%%%%%%%%%%%%5:�ѱ�������õ�Ⱦɫ��holdBestChromosome���뵽Ⱦɫ��Ⱥ��
    order=round(rand(1)*chromosomeSum);
    if order==0
        order=1;
    end
    fatherChromosomeGroup(order,:)=holdBestChromosome;
    functionError(order)=holdLeastFunctionError;
    
    %%%%%%%%%%%%%%%6:Ϊÿһ��Ⱦɫ��(�����ܽ�����)����һ������(�ؼ�����)
    %%%%%%%%%%%%%%%�õ�Ⱦɫ����ʸߣ����ĸ��ʵ͡��������functionError�������
    [p,trueP]=chromosomeProbability(functionError);
    if trueP =='Fail'
        '���ܽ����ز���Ӧ���̣������¿�ʼ'
        return%��������
    end
    %%%%%%%%%%%%%%%7�����ո���ɸѡȾɫ��(�ؼ�����)
    %fa=bin2dec(fatherChromosomeGroup)%��ʾ��Ⱦɫ��
    %�Ӹ�Ⱦ����ѡ������Ⱦɫ��
    %selecteChromosomeGroup=selecteChromosome(fatherChromosomeGroup,p);
    %%%%%%%%%%%%%%%8��Ⱦɫ���ӽ�(�ؼ�����)
    %sle=bin2dec(selecteChromosomeGroup)%��ʾѡ������Ľ�����(Ⱦɫ��)
    %�ø���ɸѡ����Ⱦɫ��selecteChromosomeGroup�����ӽ��������Ӵ�Ⱦɫ��
    %sonChromosomeGroup=crossChromosome(selecteChromosomeGroup,2);
    %���ø���ɸѡ����Ⱦɫ��selecteChromosomeGroup�����ӽ�����ֱ������һ��(����)��
    sonChromosomeGroup=crossChromosome(fatherChromosomeGroup,2);
    %sonChromosomeGroup=immunity(fatherChromosomeGroup,holdBestChromosome,3);
    %��������ֵ�����Ⱦɫ����
    sonChromosomeGroup=immunity(sonChromosomeGroup,holdBestChromosome,3);
    %cro=bin2dec(sonChromosomeGroup)%��ʾ�ӽ�����Ӵ�Ⱦɫ��
    sonChromosomeGroup=checkSequence(sonChromosomeGroup,solutionN);%����ӽ����Ⱦɫ���Ƿ�Խ��
    %%%%%%%%%%%%%%%9������
    %���ӽ�ֱ�ӱ���
    %fatherChromosomeGroup=varianceCh(fatherChromosomeGroup,0.1,solutionN);
    %�ӽ������
    fatherChromosomeGroup=varianceCh(sonChromosomeGroup,0.5,solutionN);
    fatherChromosomeGroup=checkSequence(fatherChromosomeGroup,solutionN);%��������Ⱦɫ���Ƿ�Խ��
end
