function  [Labels, DecisionValue]= SVMClass(Samples, AlphaY, SVs, Bias, Parameters, nSV, nLabel)
% Usages:
%  [Labels, DecisionValue]= SVMClass(Samples, AlphaY, SVs, Bias);
%  [Labels, DecisionValue]= SVMClass(Samples, AlphaY, SVs, Bias, Parameters);
%     Note that the above two formats are only valid for 2-class problem, it is implemented here to make this version 
%      to be compatabible with the previous version of OSU SVM ToolBox.
%  [Labels, DecisionValue]= SVMClass(Samples, AlphaY, SVs, Bias, Parameters, nSV, nLabel);
%
% DESCRIPTION:
%    Classify a group of input patterns given a trained svm classifier.
%    In fact, this function is used to do the input parameter checking, and it 
%    depends on a mex file, mexSVMClass, to implement the algorithm.
%
% Inputs:
%    Samples    - Samples to classify, MxN, (a row of column vectors);
%    AlphaY     - Alpha * Y, where Alpha is the non-zero Lagrange Coefficients, and
%                    Y is the corresponding Labels, (L-1) x sum(nSV);
%                All the AlphaYs are organized as follows: (pretty fuzzy !)
%      				classifier between class i and j: coefficients with
%			  	         i are in AlphaY(j-1, start_Pos_of_i:(start_Pos_of_i+1)-1),
%				         j are in AlphaY(i, start_Pos_of_j:(start_Pos_of_j+1)-1)
%    SVs        - Support Vectors. (Sample corresponding the non-zero Alpha), M x sum(nSV),
%                All the SVs are stored in the format as follows:
%                 [SVs from Class 1, SVs from Class 2, ... SVs from Class L];
%    Bias       - Bias of all the 2-class classifier(s), 1 x L*(L-1)/2;
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
%                         function. (default: 1)
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
%    Labels      -  Predicated Labels, 1xN. 
%    DecisionValue  -  the output of the decision function (only meaningful for 2-class problem), 1xN;
%
% By Junshui Ma, and Yi Zhao (02/15/2002)
%
if (nargin < 4 | nargin > 7)
   disp(' Incorrect number of input variables.');
   help SVMClass;
   return
end

if (nargin >= 5) 
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
if (nargin <= 5)  
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


FakeLabels = ones(1,spN); % this is a set of phony labels to feed the mex file.

% call the mex file
if (nargin == 4)
    [ClassRate, DecisionValue, Ns, ConfMatrix, Labels]= mexSVMClass(Samples, FakeLabels, AlphaY, SVs, Bias);
elseif (nargin == 5)
    [ClassRate, DecisionValue, Ns, ConfMatrix, Labels]= mexSVMClass(Samples, FakeLabels, AlphaY, SVs, Bias,Parameters);
elseif (nargin == 7)
    [ClassRate, DecisionValue, Ns, ConfMatrix, Labels]= mexSVMClass(Samples, FakeLabels, AlphaY, SVs, Bias,Parameters, nSV, nLabel);
end

 