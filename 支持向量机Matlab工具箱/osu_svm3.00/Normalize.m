function [Normalized] = Normalize(InputMat)
%function [Normlized] = Normalize(InputMat)
% Normalize the columns of matrix InputMat
Norm2 = sqrt(diag(InputMat'*InputMat));
Normalized = InputMat./(ones(size(InputMat,1),1)*Norm2');
