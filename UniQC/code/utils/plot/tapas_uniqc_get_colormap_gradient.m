function gradientColorMap = tapas_uniqc_get_colormap_gradient(rgb1, rgb2, nColors)
% creates colormap as gradient from one RGB value to another
%
%  gradientColorMap = tapas_uniqc_get_colormap_gradient(rgb1, rgb2, nColors)
%
% IN
%   rgb1    [1,3] start color 
%   rgb2    [1,3] end color
%   nColors number of colors to interpolate
%
% OUT
%   gradientColorMap [nColors,3] RGB colormap
%
% EXAMPLE
%   % gray scale colormap with 256 entries
%   tapas_uniqc_get_colormap_gradient([0 0 0], [1 1 1], 256)
%   % black to cyan colormap
%   tapas_uniqc_get_colormap_gradient([0 0 0], [0 1 1], 256)
%
%   See also
 
% Author:   Saskia Bollmann & Lars Kasper
% Created:  2019-10-03
% Copyright (C) 2019 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3. 
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.
%

 gradientColorMap = zeros(nColors, 3);
 
 for iColor = 1:3
     gradientColorMap(:,iColor) = linspace(rgb1(iColor), rgb2(iColor), nColors)';
 end