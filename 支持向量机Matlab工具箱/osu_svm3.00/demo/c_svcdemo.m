%       ------- OSU C-SVM CLASSIFIER TOOLBOX Demonstrations---
%
%       1)  Construct a linear SVM Classifier and test it
%       2)  Construct a nonlinear SVM Classifier (polynomial kernel) and test it
%       3)  Construct a nonlinear SVM Classifier (rbf kernel) and test it
%       4)  Classifier a set of input patterns
%
%       0)      Return to upper level
echo off

% OSU SVM Classifier Matlab Toolbox Demonstrations.

while 1
    demos= ['c_lindemo '
        'c_poldemo '
        'c_rbfdemo '
        'c_clademo '];
    clc
    help c_svcdemo 
    n = input('Select a demo number: ');
    if ((n <= 0) | (n > 4)) 
        break
    end
    demos = demos(n,:);
    eval(demos)
    clear
 end
 clear n demos
clc
