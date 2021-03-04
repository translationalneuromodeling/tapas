function brightenedColor = tapas_uniqc_get_brightened_color(baseColor, iStepArray, ...
    nSteps, maxBright)
% brightens color for given indices by predefined number of steps
%
%  brightenedColor = tapas_uniqc_get_brightened_color(baseColor, iStepArray, nSteps, ...
%                       maxBright)
%
% IN
%   baseColor       [1,3] RGB color
%   iStepArray    index of current interleaf / vector of interleaf
%                   indices
%   nSteps    number of total interleaves
%   maxBright       0...1 determines how white color of last interleaf
%                   1 = white; 0 = like 1st interleaf;
%                   default: 0.5
%   
% OUT
%   brightenedColor  [numel(iStepArray),3] RGB color, darker for higher-index interleaves
% 
% EXAMPLE
%   tapas_uniqc_get_brightened_color
%
%   See also

% Author: Lars Kasper
% Created: 2014-11-24
% Copyright (C) 2014 Institute for Biomedical Engineering, ETH/Uni Zurich.

if nargin < 4
    maxBright = 0.7;
end

nSelectedSteps    = numel(iStepArray);
iStepArray              = reshape(iStepArray, [],1);
baseColor               = repmat(baseColor, nSelectedSteps, 1);

 brightenedColor = baseColor + ...
     (1-baseColor).*...
     repmat((iStepArray-1)/(nSteps)*maxBright, 1,3);
