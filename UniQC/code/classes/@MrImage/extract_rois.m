function this = extract_rois(this, maskArray, keepExistingRois)
% Extracts data from this image for voxels within rois given by mask images
%
%   Y = MrImage()
%   Y.extract_rois(maskArray, keepExistingRois)
%
% This is a method of class MrImage.
%
% IN
%   maskArray           (cell array of) binary MrImage
%   keepExistingRois    if true, new rois will be concatenated to old ones
% OUT
%
% EXAMPLE
%   extract_rois
%
%   See also MrImage

% Author:   Saskia Klein & Lars Kasper
% Created:  2014-07-28
% Copyright (C) 2014 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public Licence (GPL), version 3. 
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.


if nargin < 3
    keepExistingRois = true;
end

if ~keepExistingRois
    this.rois = {};
end

if ~iscell(maskArray)
    maskArray = {maskArray};
end

nImages = numel(maskArray);

for iImage = 1:nImages
    this.rois{end+1,1} = MrRoi(this, maskArray{iImage});
end