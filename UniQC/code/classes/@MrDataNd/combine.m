function dataNdCombined = combine(this, dataNdArray, combineDims, tolerance)
% combines multiple n-dim. datasets into a single one along specified
% dimensions. Makes sure data is sorted into right place according to
% different dimInfos
%
% NOTE: inverse operation of MrDataNd.split
%
%   Y = MrDataNd()
%   dataNdCombined = Y.combine(dataNdArray, combineDims, tolerance)
%
% This is a method of class MrDataNd.
%
% IN
%   dataNdArray     cell(nDatasets,1) of MrDataNd to be combined
%   combineDims     [1, nCombineDims] vector of dim indices to be combined
%                       OR
%                   cell(1, nCombineDims) of dimLabels to be combined
%                   NOTE: If specified dimLabels do not exist, new
%                   dimensions are created with these names and default
%                   samplingPoints (1:nDatasets)
%                   default: all singleton dimensions (i.e. dims with one
%                   sample only within each individual dimInfo)
%                   NOTE: if a non-singleton dimension is given, images are
%                         concatenated along this dimension
%
%   tolerance                   dimInfos are only combined, if their
%                               information is equal for all but the
%                               combineDims (because only one
%                               representation is retained for those,
%                               usually from the first of the dimInfos).
%                               However, sometimes numerical precision,
%                               e.g., rounding errors, preclude the
%                               combination. Then you can increase this
%                               tolerance;
%                               default: single precision (eps('single')
%                               ~1.2e-7)
% OUT
%
% EXAMPLE
%   combine
%
%   See also MrDataNd MrDimInfo.combine MrDataNd.split

% Author:   Lars Kasper
% Created:  2018-05-16
% Copyright (C) 2018 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

if nargin < 4
    tolerance = eps('single');
end

% create new
dataNdCombined = this.copyobj();

%% 1) dimInfoCombined = Y.combine(dimInfoArray, combineDims)
doCombineSingletonDims = nargin < 3;
if doCombineSingletonDims
    indSplitDims = this.dimInfo.get_singleton_dimensions();
    combineDims = this.dimInfo.dimLabels(indSplitDims);
else
    % for 1-dim case, make cell
    if ~iscell(combineDims) && isstr(combineDims)
        combineDims = {combineDims};
    end
end


%% Create dimInfoArray from all data and combine it first
dimInfoArray = cellfun(@(x) x.dimInfo.copyobj(), dataNdArray, 'UniformOutput', false);

% if dimLabels not existing previously, add as new dimensions
for iDim = 1:numel(combineDims)
    combineDim = combineDims{iDim};
    % new dimension that did not exist in dimInfo
    if isempty(this.dimInfo.get_dim_index(combineDim))
        cellfun(@(x,y) x.add_dims(combineDim, 'units', 'nil', ...
            'samplingPoints', y), dimInfoArray, ...
            num2cell(1:size(dataNdArray, iDim))', ...
            'UniformOutput', false);
    end
end

[dimInfoCombined, indSamplingPointCombined] = dimInfoArray{1}.combine(...
    dimInfoArray, combineDims, tolerance);

%% Loop over all splits dataNd and put data into right place, as defined by combined DimInfo
if ~isempty(combineDims) % otherwise, nothing to do hear
    % dimInfo sampling points
    indSplitDims        = dimInfoArray{1}.get_dim_index(combineDims);
    nSplits             = numel(dataNdArray);
    dataMatrixCombined  = nan(dimInfoCombined.nSamples);
    for iSplit = 1:nSplits
        % write out indices to be filled in final array, e.g. tempData(:,:,sli, dyn)
        % would be {':', ':', sli, dyn}
        index = repmat({':'}, 1, dimInfoCombined.nDims);
        index(indSplitDims) = indSamplingPointCombined(iSplit,:);
        dataMatrixCombined(index{:}) = dataNdArray{iSplit}.data;
    end
    
    
    %% assemble the output object
    dataNdCombined.dimInfo = dimInfoCombined;
    dataNdCombined.data = dataMatrixCombined;
end
end