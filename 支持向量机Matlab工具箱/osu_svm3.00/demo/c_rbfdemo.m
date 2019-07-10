echo off
% RBFDEMO demonstration for using nonlinear SVM classifier 
% with a RBF kernel.
echo on; 

clc
% RBFDEMO demonstration for using nonlinear SVM classifier 
% with a RBF kernel.
%##########################################################################
%
%   This is a demonstration script-file for contructing and 
%     testing a nonlinear  SVM-based classifier 
%     (with a RBF kernel) using OSU SVM CLASSIFIER TOOLBOX. 
%   Note that the form of the RBF kernel is 
%               exp(-Gamma*|X(:,i)-X(:,j)|^2)
%
%##########################################################################

pause % Strike any key to continue (Note: use Ctrl-C to abort)

clc
%##########################################################################
%
%   Load the training data and examine the dimensionity of the data
%
%##########################################################################
pause % Strike any key to continue 

% load the training data
clear all
load DemoData_train

pause % Strike any key to continue 

% take a look at the data, and please pay attention to the dimensions 
% of the input data 
who

size(Labels) 
size(Samples)

pause % Strike any key to continue 

clc
%##########################################################################
%
%   Construct a nonlinear SVM classifier (with RBF kernel) 
%     using the training data
%   Note that the form of the RBF kernel is 
%     exp(-Gamma*|X(:,i)-X(:,j)|^2)
%
%##########################################################################
pause % Strike any key to continue 

% set the value of Gamma if you don't want use its default value, 
% which is 1.
Gamma = 2;

% By using this format, the default values of C, Epsilon, CacheSize
% are used. That is, C=1, Epsilon=0.001, and CacheSize=45MB
[AlphaY, SVs, Bias, Parameters, nSV, nLabel] = RbfSVC(Samples, Labels, Gamma);


% End of the SVM classifier construction 
%
% The resultant SVM classifier is jointly determined by 
%  "AlphaY", "SVs", "Bias", "Parameters", and "Ns".
%

pause % Strike any key to continue 

% Save the constructed nonlinear SVM classifier 
save SVMClassifier AlphaY SVs Bias Parameters nSV nLabel;

pause % Strike any key to continue 


clc
%##########################################################################
%
%   Test the constructed nonlinear SVM Classifier
%
%##########################################################################
pause % Strike any key to continue 

% Load the constructed nonlinear SVM classifier
clear all
load SVMClassifier

pause % Strike any key to continue 

% have a look at the variables determining the SVM classifier
who

pause % Strike any key to continue 

% load test data
load DemoData_test

pause % Strike any key to continue 

% Test the constructed SVM classifier using the test data
% begin testing ...
[ClassRate, DecisionValue, Ns, ConfMatrix, PreLabels]= SVMTest(Samples, Labels, AlphaY, SVs, Bias,Parameters, nSV, nLabel);
% end of the testing

pause % Strike any key to continue 

% The resultant confusion matrix of this 4-class classification problem is:
ConfMatrix

pause % Strike any key to continue 


echo off