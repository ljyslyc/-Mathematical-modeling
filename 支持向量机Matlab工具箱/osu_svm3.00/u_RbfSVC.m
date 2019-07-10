function [AlphaY, SVs, Bias, Parameters, nSV, nLabel] = u_RbfSVC(Samples, Labels, Gamma, C)
% USAGES: 
%    [AlphaY, SVs, Bias, Parameters, nSV, nLabel] = u_RbfSVC(Samples, Labels)
%    [AlphaY, SVs, Bias, Parameters, nSV, nLabel] = u_RbfSVC(Samples, Labels, Gamma)
%    [AlphaY, SVs, Bias, Parameters, nSV, nLabel] = u_RbfSVC(Samples, Labels, Gamma, C)
%
% DESCRIPTION: 
%   Construct a non-linear SVM classifier with a radial based kernel, or Guassian kernel, 
%     from the training Samples and Labels
%
% INPUTS:
%   Samples: all the training patterns. (a row of column vectors)
%   Lables: the corresponding class labels for the training patterns in Samples, (a row vector)
%   Gamma: parameters of the radial based kernel, which has the form
%            of (exp(-Gamma*|X(:,i)-X(:,j)|^2)). (default 1)
%   C: Cost of the constrain violation  (default 1)
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

if (nargin < 2) & (nargin > 4)
   disp(' Incorrect number of input variables.\n');
   help RbfSVC;
   return;
else
   if (nargin == 2)
       Parameters = [2 1 1 1 1 45 0.001 1];
       [AlphaY, SVs, Bias, Parameters, nSV, nLabel] = SVMTrain(Samples, Labels, Parameters);
   elseif  (nargin == 3)
       Parameters = [2 1 Gamma 1 1 45 0.001 1];
       [AlphaY, SVs, Bias, Parameters, nSV, nLabel] = SVMTrain(Samples, Labels, Parameters);
   elseif  (nargin == 4)
       Parameters = [2 1 Gamma 1 C 45 0.001 1];
       [AlphaY, SVs, Bias, Parameters, nSV, nLabel] = SVMTrain(Samples, Labels, Parameters);
   end
end
