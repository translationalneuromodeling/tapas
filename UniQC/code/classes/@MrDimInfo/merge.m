function [mergedDimInfo, commonArray, newDimLabel] = ...
    merge(this, mergeDims, varargin)
% Merges multiple dimensions into one dimensions
%
%   Y = MrDimInfo()
%   Y.merge(inputs)
%
% This is a method of class MrDimInfo.
%
% IN
%   mergeDims       which dims should be merged into one dim
%                   can be numeric [1,4] or character ['coil', 't']
%                   Note: The merged dim is always the last dim. Use
%                   permute to change order.
%
%   varargin        prop/val pairs to describe the new dimension
%                   including resolutions, ranges, dimLabels, units,
%                   samplingPoints, samplingWidths
%                   defaults:   samplingPoints = 1:nSamples
%                               dimLabels = [oldDimLabels(1), '_', ...
%                               '_', olDimLabels(end)]
%                               units = ''
%
%
% OUT
%
% EXAMPLE
%   newDimInfo = dimInfo.merge({'coil', 'echo'})
%
%   See also MrDimInfo

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2019-12-23
% Copyright (C) 2019 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

% create selection object that contains the to-be-merged dims and the new
% default properties (dimLabels and nSamples)
for nSelect = 1:numel(mergeDims)
    if ischar(mergeDims{nSelect})
        dimIndex = this.get_dim_index(mergeDims{nSelect});
        dimLabel = mergeDims{nSelect};
    elseif isnumeric(mergeDims{nSelect})
        dimIndex = mergeDims{nSelect};
        dimLabel = this.dimLabels(dimIndex);
    else
        error('tapas:uniqc:MrDimInfo:InvalidMergeDimension', ...
            'Invalid mergeDims specifier. Allowed are dimLabels or dimIndices.');
    end
    commonArray.(dimLabel) = 1:this.nSamples(dimIndex);
    if nSelect == 1
        newDimLabel = dimLabel;
        newNSamples = this.nSamples(dimIndex);
    else
        newDimLabel = [newDimLabel, '_', dimLabel];
        newNSamples = newNSamples * this.nSamples(dimIndex);
    end   
end

selectInput = commonArray;

% use inverted selection to get all dims that are kept
selectInput.invert = 1;
selectInput.removeDims = 1;

mergedDimInfo = this.select(selectInput);

% add new dims
mergedDimInfo.add_dims(mergedDimInfo.nDims + 1, 'dimLabels', newDimLabel, ...
    'nSamples', newNSamples, 'units', '');

% make changes to new dim
mergedDimInfo.set_dims(mergedDimInfo.nDims, varargin{:});

% update new dim label
newDimLabel = mergedDimInfo.dimLabels{mergedDimInfo.nDims};
end

