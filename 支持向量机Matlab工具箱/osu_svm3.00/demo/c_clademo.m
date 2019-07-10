echo off
% CLADEMO demonstration for using a contructed SVM classifier to classify 
% input patterns
echo on; 
%
%
% NOTICE: please first run any of the first three demonstrations before
%         running this, because this demonstrate needs the results obtained
%         from any of the previous demonstrations.
%

pause

clc
% CLADEMO demonstration for using a contructed SVM classifier to classify 
% input patterns
%##########################################################################
%
%   This is a demonstration script-file for classifying the
%   input patterns using a constructed SVM classifier 
%
%##########################################################################

pause % Strike any key to continue (Note: use Ctrl-C to abort)

clc
%##########################################################################
%
% Classify input patterns using the constructed SVM Classifier
%
%##########################################################################
pause % Strike any key to continue 

% Load the constructed SVM classifier
clear all
load SVMClassifier

pause % Strike any key to continue 

% have a look at the variables determining the SVM classifier
who

pause % Strike any key to continue 

% load the patterns
load DemoData_class

pause % Strike any key to continue 


% Classify input patterns using the constructed nonlinear SVM Classifier
% begin classification ......
[Labels, DecisionValue]= SVMClass(Samples, AlphaY, SVs, Bias, Parameters, nSV, nLabel);
% end of the classification

pause % Strike any key to continue 

% Compare the resultant labels with the true labels of the data
plot(1:length(TrueLabels),TrueLabels,'b-',1:length(TrueLabels),Labels,'r.');
ylabel('Class Index');
xlabel('Pattern Index');
legend('True Labels','Resultant Labels',0);

pause % Strike any key to continue 


echo off