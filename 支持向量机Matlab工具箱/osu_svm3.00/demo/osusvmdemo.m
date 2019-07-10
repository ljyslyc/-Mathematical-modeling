%       ------- OSU SVM CLASSIFIER TOOLBOX Demonstrations---
%
%       1)  Demonstrations of using C-SVM Classifers.
%       2)  Demonstrations of using u-SVM Classifiers
%       3)  Demonstration of uisng 1-SVM 
%
%       0)      Quit
echo off

% OSU SVM Classifier Matlab Toolbox Demonstrations.

while 1
    demos= ['c_svcdemo   '
            'u_svcdemo   '
            'one_rbfdemo '];
    clc
    help osusvmdemo 
    n = input('Select a demo number: ');
    if ((n <= 0) | (n > 3)) 
        break
    end
    demos = demos(n,:);
    eval(demos)
    clear
 end
 clear n demos
clc
