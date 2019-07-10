function [ClassRate, DecisionValue, Ns, ConfMatrix, PreLabels]= SVMTest(Samples, Labels, AlphaY, SVs, Bias,Parameters, nSV, nLabel)
% Usages:
%  [ClassRate, DecisionValue, Ns, ConfMatrix, PreLabels]= SVMTest(Samples, Labels, AlphaY, SVs, Bias)
%  [ClassRate, DecisionValue, Ns, ConfMatrix, PreLabels]= SVMTest(Samples, Labels, AlphaY, SVs, Bias, Parameters)
%     Note that the above two formats are only valid for 2-class problem, it is implemented here to make this version 
%      to be compatabible with the previous version of OSU SVM ToolBox.
%  [ClassRate, DecisionValue, Ns, ConfMatrix, PreLabels]= SVMTest(Samples, Labels, AlphaY, SVs, Bias, Parameters, nSV, nLabel)
%
% DESCRIPTION:
%    Test the performance of a trained SVM classifier by a group of input patterns
%    with their true class labels given.
%    In fact, this function is used to do the input parameter checking, and it 
%    depends on a mex file, mexSVMClass, to implement the algorithm.
%
% Inputs:
%    Samples    - testing samples, MxN, (a row of column vectors);
%    Labels     - labels of testing samples, 1xN, (a row vector);
%    AlphaY    - Alpha * Y, where Alpha is the non-zero Lagrange Coefficients, and
%                    Y is the corresponding Labels, (L-1) x sum(nSV);
%                All the AlphaYs are organized as follows: (pretty fuzzy !)
%      				classifier between class i and j: coefficients with
%			  	         i are in AlphaY(j-1, start_Pos_of_i:(start_Pos_of_i+1)-1),
%				         j are in AlphaY(i, start_Pos_of_j:(start_Pos_of_j+1)-1)
%    SVs       - Support Vectors. (Sample corresponding the non-zero Alpha), M x sum(nSV),
%                All the SVs are stored in the format as follows:
%                 [SVs from Class 1, SVs from Class 2, ... SVs from Class L];
%    Bias      - Bias of all the 2-class classifier(s), 1 x L*(L-1)/2;
%    Parameters - the paramters required by the training algorithm (a <=11-element row vector);
%     +------------------------------------------------------------------
%     |Kernel Type| Degree | Gamma | Coefficient | C |Cache size|epsilon| 
%     +------------------------------------------------------------------
%       ----------------------------------------------+
%       | SVM type | nu | loss toleration | shrinking |
%       ----------------------------------------------+
%            where Kernel Type: (default: 2) 
%                     0 --- Linear
%                     1 --- Polynomial: (Gamma*<X(:,i),X(:,j)>+Coefficient)^Degree
%                     2 --- RBF: (exp(-Gamma*|X(:,i)-X(:,j)|^2)) 
%                     3 --- Sigmoid: tanh(Gamma*<X(:,i),X(:,j)>+Coefficient)
%                  Degree: default 3
%                  Gamma: If the input value is zero, Gamma will be set defautly as
%                         1/(max_pattern_dimension) in the function. If the input
%                         value is non-zero, Gamma will remain unchanged in the 
%                         function. (default: 0 or 1/M)
%                  Coefficient: default 0
%                  C: Cost of constrain violation for C-SVC, epsilon-SVR, and nu-SVR (default 1)
%                  Cache Size: Space to hold the elements of K(<X(:,i),X(:,j)>) matrix (default 40MB)
%                  epsilon: tolerance of termination criterion (default: 0.001)
%                  SVM Type: (default: 0)
%                     0 --- c-SVC 
%                     1 --- nu-SVC
%                     2 --- one-class SVM
%                     3 --- epsilon-SVR 
%                     4 --- nu-SVR
%                  nu: nu of nu-SVC, one-class SVM, and nu-SVR (default: 0.5)
%                  loss tolerance: epsilon in loss function of epsilon-SVR (default: 0.1)
%                  shrinking: whether to use the shrinking heuristics, 0 or 1 (default: 1)
%    nSV       -  numbers of SVs in each class, 1xL;
%    nLabel    -  Labels of each class, 1xL.
%
% Outputs:  
%    ClassRate      -  Classification rate, 1x1;
%    DecisionValue  -  the output of the decision function (only meaningful for 2-class problem), 1xN;
%    Ns             -  number of samples in each class, 1x(L+1), or 1xL;
%                       Note that the last element is for the Samples that are not in any
%                         classes in the training set.
%    ConfMatrix     -  Confusion Matrix, (L+1)x(L+1), or LxL, where ConfMatrix(i,j) = P(X in j| X in i);
%                       Note that when (L+1)x(L+1), the last row and the last column are for the Samples 
%                       that are not in any classes in the training set.
%    PreLabels      -  Predicated Labels, 1xN. 
%
% By Junshui Ma, and Yi Zhao (02/15/2002)
%

if (nargin < 5 | nargin > 8)
   disp(' Error: Incorrect number of input variables.');
   help SVMTest;
   return
end

[minLabel, I]=min(Labels);
[maxLabel, I]=max(Labels);
if ((minLabel ~= -1) | (maxLabel ~= 1))
    if (nargin < 8)
        disp(' Error: The sample labels are not in {-1,1}, However, you need to input ''nLabel'' to support speical labels.');
        return
    end
end
    

if (nargin >= 6) 
    [prM prN]= size(Parameters);
    if (prM ~= 1 & prN~=1)
        disp(' Error: ''Parameters'' should be a row vector.');
        return
    elseif (prM~= 1)
        Parameters = Parameters';
        [prM prN]= size(Parameters);
    end
    if (Parameters(1)>3) & (Parameters(1) < 0)
        disp(' Error: this program only supports 4 types of kernel functions.');
        return
    end
    if (prN >=8)
        if (Parameters(8)>4) & (Parameters(8) <0)
           disp(' Error: this program only supports 5 types of SVMs.');
           return
        end
    end
end



[alM alN] = size(AlphaY);
if (nargin <= 6)  
    [r c] = size(Bias);
    if (r~=1 | c~=1)
        disp(' Error: Your SVM classifier seems a multiclass classifier. However, you need to input ''nSV'' and ''nLabel'' to support multiclass problem.');
        return
    end    
    if (alM > 1)
        disp(' Error: Your SVM classifier seems a multiclass classifier. However, you need to input ''nSV'' and ''nLabel'' to support multiclass problem.');
        return
    end 
end

[spM spN]=size(Samples);
[svM svN]=size(SVs);

if svM ~= spM
   disp(' Error: ''SVs'' should have the same feature dimension as ''Samples''.');
   return;
end

if svN ~= alN
   disp(' Error: number of ''SVs'' should be the same as the colmun number of ''AlphaY''.');
   return;
end


% call the mex file
if (nargin == 5)
    [ClassRate, DecisionValue, Ns, ConfMatrix, PreLabels]= mexSVMClass(Samples, Labels, AlphaY, SVs, Bias);
elseif (nargin == 6)
    [ClassRate, DecisionValue, Ns, ConfMatrix, PreLabels]= mexSVMClass(Samples, Labels, AlphaY, SVs, Bias,Parameters);
elseif (nargin == 8)
    [ClassRate, DecisionValue, Ns, ConfMatrix, PreLabels]= mexSVMClass(Samples, Labels, AlphaY, SVs, Bias,Parameters, nSV, nLabel);
end

% if these is no extra class in the testing samples, 
% remove that last column and row in ConfMatrix
if (ConfMatrix(end, end) == 1) 
    ConfMatrix = ConfMatrix(1:end-1,1:end-1);
    Ns = Ns(1:end-1);
end

 