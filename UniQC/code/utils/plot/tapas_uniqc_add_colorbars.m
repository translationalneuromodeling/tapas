function hColorbarAxes = tapas_uniqc_add_colorbars(hax, imageColorMaps, imageRanges, ...
    imageNames)
% Creates Colorbars next to image aixs for all given colormaps/value ranges
%
% NOTE: The plotted colorbars are actual normal axes with plots, NOT of
%       class colorbar
%   output = tapas_uniqc_add_colorbars(input)
%
% IN
%   hax             handle of axes with image where colorbars shall be created
%   imageColorMaps  cell(nColorMaps,1) of [nColors,3] matrices, i.e. color
%                   maps used in image
%   imageRanges     cell(nColorMaps,1) of [minValue, maxValue] in image
%                   corresponding to colormap
%   imageNames      cell(nColorMaps,1) of strings labelling each colormap
%
% OUT
%   hColorbarAxes   cell(nColorMaps, 1) of axes handles for colorbars
%
% EXAMPLE
%   tapas_uniqc_add_colorbars
%
%   See also

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2014-11-27
% Copyright (C) 2014 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public Licence (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.


nColorMaps          = numel(imageColorMaps);
positionImageAxis   = get(hax, 'Position');
dxColorBar          = 0.01;
startColorBar       = positionImageAxis(1)+positionImageAxis(3) ...
    - dxColorBar*1;

positionAxes        = [startColorBar, positionImageAxis(2), ...
    dxColorBar, positionImageAxis(4)]; 

% add colorbars for all maps
for iColorMap = 1:nColorMaps
    
    % create new axis
    positionAxes(1) = positionAxes(1) + dxColorBar*3;
    hax             = axes('Position', positionAxes);
    
    % Plot colorbar as image
    nColors = size(imageColorMaps{iColorMap},1);
    imagesc(permute(imageColorMaps{iColorMap}, [1 3 2]));
    axis xy;
    set(hax, 'YTick', []);
    set(hax, 'XTick', [])
    
    dy = (imageRanges{iColorMap}(2) - imageRanges{iColorMap}(1))/nColors;
    
    % set min value as Xlabel
    xlabel(sprintf('%.0f', imageRanges{iColorMap}(1)));
    hlx = get(hax, 'XLabel');
    set(hlx, 'Units', 'normalized');
    set(hlx, 'Position', [0.5, -0.01]);
    
    % set max value as title 
    title(sprintf('%.0f', imageRanges{iColorMap}(2)));
    hTitle = get(hax, 'Title'); 
    set(hTitle, 'FontWeight', 'normal');
    
    % set title as ylabel
    ylabel(tapas_uniqc_str2label(imageNames{iColorMap}));
    hly = get(hax, 'YLabel');
    positionLabel = get(hly, 'Position');
    set(hly, 'Position', [0, positionLabel(2)]);
 
end