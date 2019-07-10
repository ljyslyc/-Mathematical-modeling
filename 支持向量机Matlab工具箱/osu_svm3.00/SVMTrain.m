function [AlphaY, SVs, Bias, Parameters, nSV, nLabel] = SVMTrain(Samples, Labels, Parameters, Weight)
% Usages:
% [AlphaY, SVs, Bias, Parameters, nSV, nLabel] = SVMTrain(Samples, Labels)
% [AlphaY, SVs, Bias, Parameters, nSV, nLabel] = SVMTrain(Samples, Labels, Parameters)
% [AlphaY, SVs, Bias, Parameters, nSV, nLabel] = SVMTrain(Samples, Labels, Parameters, Weight)
%
% DESCRIPTION:
% Construct a SVM classifier. 
% In fact, This function is used to do the input parameter checking, and it calls a mex file, mexSVMTrain, to 
% implement the algorithm.
%
% Inputs:
%    Samples    - training samples, MxN, (a row of column vectors);
%    Labels     - labels of training samples, 1xN, (a row vector);
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
%    Weight     - a row vector or scalar, C of class i is weight(i)*C in C-SVC (default: all 1's);
%
% Outputs:  
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
%    Parameters -  Output parameters used in training;
%    nSV       -  numbers of SVs in each class, 1xL;
%    nLabel    -  Labels of each class, 1xL.
%
% By Junshui Ma, and Yi Zhao (02/15/2002)
%

if (nargin < 2 | nargin > 4)
   disp(' Error: Incorrect number of input variables.');
   help SVMTrain;
   return
end

if (nargin >= 3) 
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
    if (prN >=9)    
        if ((Parameters(8)==1) | (Parameters(8) == 2) | (Parameters(8) == 4)) & (Parameters(9) >= 1)
           disp(' Error: the nu for nu-SVC, one-class SVM, and nu-SVR should be less than 1 and bigger than 0');
           return
        end        
    end
end

[spM spN]=size(Samples);
[lbM lbN]=size(Labels);
if lbM ~= 1
   disp(' Error: ''Labels'' should be a row vector.');
   return
end
if spN ~= lbN
   disp(' Error: the number of training samples is different from that of their labels.');
   return
end

% call the mex file
if (nargin == 2)
    [AlphaY, SVs, Bias, Parameters, nSV, nLabel] = mexSVMTrain(Samples, Labels);
elseif (nargin == 3)
    [AlphaY, SVs, Bias, Parameters, nSV, nLabel] = mexSVMTrain(Samples, Labels, Parameters);
elseif (nargin == 4)
    [AlphaY, SVs, Bias, Parameters, nSV, nLabel] = mexSVMTrain(Samples, Labels, Parameters, Weight);
end


 