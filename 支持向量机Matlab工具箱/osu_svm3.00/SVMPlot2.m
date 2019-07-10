function SVMPlot(AlphaY, SVs, Bias, Parameters,Samples, Labels, aspect,mag,xaxis,yaxis,input)
% USAGE:
%   SVMPlot(AlphaY, SVs, Bias, Parameters,Samples, Labels);
%   SVMPlot(AlphaY, SVs, Bias, Parameters,Samples, Labels, aspect,mag,xaxis,yaxis,input)
%
% DESCRIPTION:
%  Plot a 2-class Support Vector classifier with some labelled samples.
%
% INPUTS:
%  AlphaY: Alpha * Y, where Alpha is the non-zero Lagrange Coefficients
%                    Y is the corresponding labels.
%  SVs: support vectors. That is, the patterns corresponding the non-zero
%       Alphas.
%  Bias: the bias in the decision function, which is AlphaY*Kernel(SVs',x)-Bias.
%  Parameters:the paramters required by the training algorithm. 
%             (a 10-element row vector)
%            +-----------------------------------------------------------------
%            |Kernel Type| Degree | Gamma | Coefficient | C |Cache size|epsilon| 
%            +-----------------------------------------------------------------
%				 -------------------------------------------+
%            | SVM Type |	nu (nu-svm) |   Percentage   |				
%				 -------------------------------------------+
%            where Kernel Type:
%                   0 --- Linear
%                   1 --- Polynomial: (Gamma*<X(:,i),X(:,j)>+Coefficient)^Degree
%                   2 --- RBF: (exp(-Gamma*|X(:,i)-X(:,j)|^2))
%                   3 --- Sigmoid: tanh(Gamma*<X(:,i),X(:,j)>+Coefficient)
%                  Gamma: If the input value is zero, Gamma will be set defautly as
%                         1/(max_pattern_dimension) in the function. If the input
%                         value is non-zero, Gamma will remain unchanged in the 
%                         function.
%                  C: Cost of the constrain violation (for C-SVC & C-SVR), or the 
%                     nu for the nu-svm and 1-svm in the second stage. This 
%                     nu should smaller than the nu defined in Parameter(9)
%                  Cache Size: as the buffer to hold the <X(:,i),X(:,j)> (in MB)
%                  epsilon: tolerance of termination criterion
%                  SVM Type: (Recommend only using 0, which is c-SVM classifier)
%                   0 --- c-SVM classifier
%                   1 --- nu-SVM classifier
%                  nu: the nu of nu-SVM used in the boundary finding process
%						 Percentage: the percentage of input train samples used for boundary finding.	
%  Samples: all the training patterns. (a row of column vectors)
%  Labels:  the corresponding class labels for the training patterns in Samples.
%         where Labels(i) in {1, -1}. (a row vector)
%  aspect: Aspect Ratio (default: 0 (fixed), 1 (variable))
%  mag: display magnification  (default: 0.1) (kind of side margin)
%  xaxis: xaxis input (default: 1)
%  yaxis: yaxis input (default: 2)
%  input: vector of input values (default: zeros(no_of_inputs))
%
%  Orignal Author: Steve Gunn (srg@ecs.soton.ac.uk)
%  Modified by Junshui Ma

if (nargin < 6 | nargin > 11) % check correct number of arguments
   help SVMPlot;
   return;
else
   
   [maxLabel, I] = max(Labels);
   [minLabel, I] = min(Labels);
   
   if (maxLabel ~= 1) & (minLabel ~=-1)
       if (maxLabel ~= minLabel) 
           Labels(find(Labels==maxLabel)) = 1;
           Labels(find(Labels==minLabel)) = -1;
       end    
   end
   
   epsilon = 1e-5;  
   if (nargin < 11) input = zeros(size(Samples,1),1);, end
   if (nargin < 10) yaxis = 2;, end
   if (nargin < 9) xaxis = 1;, end
   if (nargin < 8) mag = 0.1;, end
   if (nargin < 7) aspect = 0;, end
    
   % Scale the axes
   xmin = min(Samples(xaxis,:));
   xmax = max(Samples(xaxis,:)); 
   
   ymin = min(Samples(yaxis,:));
   ymax = max(Samples(yaxis,:));
   
   xa = (xmax - xmin);
   ya = (ymax - ymin);
   if (~aspect)
       if (0.75*abs(xa) < abs(ya)) 
          offadd = 0.5*(ya*4/3 - xa);, 
          xmin = xmin - offadd - mag*0.5*ya;
          xmax = xmax + offadd + mag*0.5*ya;
          ymin = ymin - mag*0.5*ya;
          ymax = ymax + mag*0.5*ya;
       else
          offadd = 0.5*(xa*3/4 - ya);, 
          xmin = xmin - mag*0.5*xa;
          xmax = xmax + mag*0.5*xa;
          ymin = ymin - offadd - mag*0.5*xa;
          ymax = ymax + offadd + mag*0.5*xa;
       end
    else
       xmin = xmin - mag*0.5*xa;
       xmax = xmax + mag*0.5*xa;
       ymin = ymin - mag*0.5*ya;
       ymax = ymax + mag*0.5*ya;
    end
    
    set(gca,'XLim',[xmin xmax],'YLim',[ymin ymax]);  

    % Plot function value

    [x,y] = meshgrid(xmin:(xmax-xmin)/50:xmax,ymin:(ymax-ymin)/50:ymax); 

	 z = Bias*ones(size(x));
    Inputs = input*ones(1,size(x,2));

    wh = waitbar(0,'Plotting...');
    for x1 = 1 : size(x,1)
       Inputs(xaxis,:) = x(x1,:);
       Inputs(yaxis,:) = y(x1,:);
       [OutLabels, z(x1,:)]= SVMClass(Inputs,AlphaY, SVs, Bias,Parameters);
       waitbar((x1)/size(x,1));
    end
    close(wh);
    
    l = (-min(min(z)) + max(max(z)))/2.0;
    sp = pcolor(x,y,z);
    shading interp;  
    set(sp,'LineStyle','none');
    set(gca,'Clim',[-l  l],'Position',[0 0 1 1]);
    axis off;
    load cmap;
    colormap(colmap);

    % Plot Training points

   hold on
   for i = 1:length(Labels)
      if (Labels(i) == 1)
        plot(Samples(xaxis,i),Samples(yaxis,i),'b+','LineWidth',4); % Class A
      else
        plot(Samples(xaxis,i),Samples(yaxis,i),'r+','LineWidth',4); % Class B
      end
   end
   for i = 1:length(AlphaY) 
      plot(SVs(xaxis,i),SVs(yaxis,i),'wo'); % Support Vector
   end
   
   % Plot Boundary contour
   hold on
   contour(x,y,z,[0 0],'w');
   if (Parameters(8) ~=2) 
      contour(x,y,z,[1 1],'k:');
      contour(x,y,z,[-1 -1],'k:');
   end   
   hold off
end    
