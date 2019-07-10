function [AlphaY, SVs, Bias, Parameters, nSV, nLabel] = u_PolySVC(Samples, Labels, Degree, C, Gamma, Coeff)
% USAGES: 
%    [AlphaY, SVs, Bias, Parameters, nSV, nLabel] = u_PolySVC(Samples, Labels)
%    [AlphaY, SVs, Bias, Parameters, nSV, nLabel] = u_PolySVC(Samples, Labels, Degree)
%    [AlphaY, SVs, Bias, Parameters, nSV, nLabel] = u_PolySVC(Samples, Labels, Degree, C)
%    [AlphaY, SVs, Bias, Parameters, nSV, nLabel] = u_PolySVC(Samples, Labels, Degree, C, Gamma, Coeff)
%
% DESCRIPTION: 
%    Construct a non-linear SVM classifier with a polynomial kernel 
%        from the training Samples and Labels
%
% INPUTS:
%     Samples: all the training patterns, (a row of column vectors)
%     Lables: the corresponding class labels for the training patterns in Samples,(a row vector)
%     Degree, Gamma, and Coeff: parameters of the polynomial kernel, which has the form
%                                of  (Gamma*<X(:,i),X(:,j)>+Coefficient)^Degree
%                 Degree ---(default: 3)
%                 Gamma  ---(default: 1)
%                 Coeff  ---(default: 1)
%     C: Cost of the constrain violation  (default: 1)
%
% OUTPUTS:
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

if (nargin < 2) & (nargin > 6)
   disp(' Incorrect number of input variables.\n');
   help PolySVC;
   return;
else
   if (nargin == 2)
       Parameters = [1 3 1 1 1 45 0.001 1];
       [AlphaY, SVs, Bias, Parameters, nSV, nLabel] = SVMTrain(Samples, Labels, Parameters);
   elseif  (nargin == 3)
       Parameters = [1 Degree 1 1 1 45 0.001 1];
       [AlphaY, SVs, Bias, Parameters, nSV, nLabel] = SVMTrain(Samples, Labels, Parameters);
   elseif  (nargin == 4)
       Parameters = [1 Degree 1 1 C 45 0.001 1];
       [AlphaY, SVs, Bias, Parameters, nSV, nLabel] = SVMTrain(Samples, Labels, Parameters);
   elseif  (nargin == 6)
       Parameters = [1 Degree Gamma Coeff C 45 0.001 1];
       [AlphaY, SVs, Bias, Parameters, nSV, nLabel] = SVMTrain(Samples, Labels, Parameters);
   end
end
