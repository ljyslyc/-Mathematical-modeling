function tbxStruct=Demos
% DEMOS    Demo List information for OSV SVM Classifier Matlab Toolbox


if nargout==0, demo toolbox; return; end

tbxStruct.Name='OSU SVM Classifier';
tbxStruct.Type='toolbox';

tbxStruct.Help= {
   ' The OSV SVM Classifier Toolbox contains            '  
   ' commands for training and testing a SVM-based classifier,'
   ' and the command for classifying patterns using the '
   ' trained SVM classifier. '  
   '                                                '};
tbxStruct.DemoList={  
   'Command Line Demos', 'osusvmdemo'};







