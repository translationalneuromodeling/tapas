function croppedImages = crop_all(this, rangeIndexArray, otherImages)
%Crops all given images to same range as this image
%
%   Y = MrDataNd()
%   Y.crop_all(rangeIndexArray, otherImages)
%
% This is a method of class MrDataNd.
%
% IN
%   rangeIndexArray     cell(1,2*nDimsToCrop) of pairs of
%                       dimLabel, [startSampleIndex endSampleIndex]
%                       referring to dimensions of reference image for
%                       cropping (this), e.g.,
%                       {'x', [10 100], 'y', [50 120], 'z', [10 30]}
%
%   otherImages         cell(nImages,1) of MrDataNd
%                           OR
%                       single MrDataNd object to be cropped
% OUT
%   croppedImages       cell(nImages+1,1) of equally cropped images,
%                       including reference image for cropping (this)
% EXAMPLE
%   crop_all
%
%   See also MrDataNd

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2019-09-06
% Copyright (C) 2019 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

if ~iscell(otherImages)
    otherImages = {otherImages};
end
nImages = numel(otherImages) + 1;

%% create
croppedImages = cell(nImages,1);

dimLabels = rangeIndexArray(1:2:end);
cropIndexRanges = rangeIndexArray(2:2:end);

%% convert index crop of this image into range and do the crop

nDims = numel(dimLabels);
selectionIndexArray = cell(1, 2*nDims);


%% convert cropping to current image and apply crop
croppedImages = [{this}; reshape(otherImages, [], 1)];
for iImage = 1:nImages
    
    %% setup selectionIndexArray for cropping, dim by dim
    for iDim = 1:nDims
        cropIndexRange = cropIndexRanges{iDim};
        dimLabel = dimLabels{iDim};
        idxDim = this.dimInfo.get_dim_index(dimLabel);
        crop = this.dimInfo.index2sample(cropIndexRange, idxDim);
        res = croppedImages{iImage}.dimInfo.resolutions;
        selectionIndexArray{2*iDim-1} = dimLabel;
        selectionIndexArray{2*iDim} = crop(1):res(idxDim):crop(2);
    end
    
    %% crop
    croppedImages{iImage} = croppedImages{iImage}.select('type', 'samples', ...
        selectionIndexArray{:});
end