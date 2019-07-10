function SVMPlot(AlphaY, SVs, Bias, Parameters,Samples, Labels, nSV, nLabel, aspect,mag,xaxis,yaxis,input)
% USAGE:
%   SVMPlot(AlphaY, SVs, Bias, Parameters,Samples, Labels, nSV, nLabel
%   SVMPlot(AlphaY, SVs, Bias, Parameters,Samples, Labels, nSV, nLabel, aspect,mag,xaxis,yaxis,input)
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
%  Samples: all the training patterns. (a row of column vectors)
%  Labels:  the corresponding class labels for the training patterns in Samples.
%         where Labels(i) in {1, -1}. (a row vector)
%  nSV       -  numbers of SVs in each class, 1xL;
%  nLabel    -  Labels of each class, 1xL.
%  aspect: Aspect Ratio (default: 0 (fixed), 1 (variable))
%  mag: display magnification  (default: 0.1) (kind of side margin)
%  xaxis: xaxis input (default: 1)
%  yaxis: yaxis input (default: 2)
%  input: vector of input values (default: zeros(no_of_inputs))
%
%  By Junshui Ma

if (nargin < 8 | nargin > 13) % check correct number of arguments
   help SVMPlot;
   return;
else
   if (nLabel > 12) 
       disp(['This function can only maximally display a 12-class problem']);
       return;
   end
   Symbol = {'bd','r<', 'b*', 'rx', 'bs', 'rv', 'b^', 'r+', 'b>', 'rp', 'bh', 'gx'};
   

   epsilon = 1e-5;  
   if (nargin < 13) input = zeros(size(Samples,1),1);, end
   %if (nargin < 13) input = mean(Samples,2);, end
   if (nargin < 12) yaxis = 2;, end
   if (nargin < 11) xaxis = 1;, end
   if (nargin < 10) mag = 0.1;, end
   if (nargin < 9) aspect = 0;, end
    
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

	z = ones(size(x));
    OutLabels = ones(size(x));
    Inputs = input*ones(1,size(x,2));

    wh = waitbar(0,'Plotting...');
    for x1 = 1 : size(x,1)
       Inputs(xaxis,:) = x(x1,:);
       Inputs(yaxis,:) = y(x1,:);
       [OutLabels(x1,:), z(x1,:)]= SVMClass(Inputs,AlphaY, SVs, Bias,Parameters, nSV, nLabel);
       waitbar((x1)/size(x,1));
    end
    close(wh);
    
    sp = pcolor(x,y,OutLabels);
    shading interp;  
    set(sp,'LineStyle','none');
    %set(gca,'Clim',[-l  l],'Position',[0 0 1 1]);
    set(gca,'Clim',[min(min(OutLabels))  max(max(OutLabels))],'Position',[0 0 1 1]);
    axis off;
    load cmap;
    colormap(colmap);

    % Plot Training points

   hold on
   for i = 1:length(Labels)
       for j=1:length(nLabel)
           if Labels(i) == nLabel(j)
               plot(Samples(xaxis,i),Samples(yaxis,i), Symbol{j});
               break;
           end
       end
   end
   
   for i = 1:length(AlphaY) 
      plot(SVs(xaxis,i),SVs(yaxis,i),'wo');    % Support Vector
   end
   
   hold off
end    
