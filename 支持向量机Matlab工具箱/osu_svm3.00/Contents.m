% OSU Support Vector Machines (SVMs) Toolbox
% version 3.00, Feb. 2002
%
% The core of this toolbox is based on Dr. Lin's Lib SVM version 2.33
% For more details, please see:
%  http://www.csie.ntu.edu.tw/~cjlin/libsvm
%
% Data Preprocessing:
%
%  Normalize: normalize all the samples to make their energy is 1
%  Scale: scale all the samples to a range, such as [-1 1]
%
% SVM classifier Trainer:
%
%   LinearSVC (u_LinearSVC) 
%           - construct a linear C-SVM (nu-SVM) classifier 
%               from training samples.
%   PolySVC (u_PolySVC)
%           - construct a non-linear C-SVM (nu-SVM) classifier 
%             with a polynomial kernel.
%   RbfSVC (u_RbfSVC)
%           - construct a non-linear C-SVM (nu-SVM) classifier 
%            with a radial based kernel, or Gaussian kernel.
%   one-RbfSVC
%           - construct a non-linear 1-SVM with a radial based 
%             kernel, or Gaussian kernel.
%
% C-SVC Tester:
%
%   SVMTest - test the performance of a trained 
%                SVM classifier%
% 
% C-SVM Classifier:
%   
%   SVMClass - classify a set of input patterns 
%                 given a trained SVM classifier%
% Low level functions: 
% (following functions are called by functions listed above)
%   SVMClass, 
%   mexSVMTrain, mexSVMClass
%
% Plot results:
%   SVMPlot2: plot out the training samples and classification boundaries of a two-class problem
%   SVMPlot: plot out the training samples and classification boundaries of a multi-class problem, 
%            However, generally speaking, the plots obtained by this function is not very attractive.
%
% Demonstration functions:
%  Demo\osusvmdemo - the main function of the command-line demonstration
%
%  Demo\c_lindemo (u_lindemo) 
%           -  demonstration for constructing and test a linear C-SVM, or nu-SVM, 
%              classifier
%  Demo\c_poldemo (u_poldemo)
%           - demonstration for constructing and test a nonlinear C-SVM, or nu-SVM, 
%              classifier with a polynomial kernel
%  Demo\c_rbfdemo (u_rbfdemo)
%           - demonstration for constructing and test a nonlinear C-SVM, or nu-SVM,
%              classifier with a RBF kernel
%  Demo\one_rbfdemo 
%           - demonstration for constructing and test a nonlinear 1-SVM 
%             with a RBF kernel
%  Demo\c_clademo (u_clademo)
%           - demonstration for classify a group of input patterns using
%             the constructed SVM classifier.
%  Demo\DemoData_train, Demo\DemoData_test, and Demo\DemoData_class - data used
%               in this demonstration. they are HRR radar signatures
%               generated from MSTAR data.
%----------------------------------------
% Authors: 
% Junshui Ma (junshui@lanl.gov), NIS-2, Los Alamos National Lab
% Yi Zhao (zhaoy@ee.eng.ohio-state.edu), EE department, Ohio State University
%
