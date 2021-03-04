function manualMask = draw_mask(this, varargin)
% Displays slices of a 3D volume to manually draw a polygonal mask on
% them and combines output into 3D mask with same image dimensions. 
%
%   Y = MrImage()
%   manualMask = Y.draw_mask('z', Inf, 't', 1)
%
% This is a method of class MrImage.
%
% IN
%   
%   z       default: Inf (all) on which mask can be drawn
%   t       default: 1 on which mask can be drawn
%
% OUT
%   manualMask      MrImage with same geometry as this image for 1st 3 
%                   dimensions 
%                   Unless number of selectedVolumes is greater than 1,
%                   manualMask will be a 3D image. 
%
% EXAMPLE
%   manualMask = Y.draw_mask('z', Inf, 't', 1)
%
%   See also MrImage

% Author:   Saskia Klein & Lars Kasper
% Created:  2014-11-13
% Copyright (C) 2014 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public Licence (GPL), version 3. 
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

defaults.z = [];
defaults.t = 1;

args = tapas_uniqc_propval(varargin, defaults);
tapas_uniqc_strip_fields(args);

if isempty(z) || any(isinf(z))
    z = 1:this.dimInfo.nSamples(3); % TODO: make it dependent on name of label?!?
end

[manualPlotData, selectionIndexArray, unusedDims] = this.select('t', t);
manualMask      = manualPlotData.copyobj();
manualMask.data = zeros(manualPlotData.dimInfo.nSamples);
for iSlice = z
    manualPlotData.select('z', iSlice).plot();
    manualMask.data(:,:, iSlice) = roipoly();
end