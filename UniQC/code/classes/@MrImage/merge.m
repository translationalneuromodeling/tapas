function [mergedImage, newDimLabel] = ...
    merge(this, mergeDims, varargin)
% Merges multiple dimensions into one dimensions
%
%   Y = MrImage()
%   mergedImage = Y.merge({'echo', 't'})
%
% This is a method of class MrImage.
%
% IN
%   mergeDims       which dims should be merged into one dim
%                   can be numeric [1,4] or character {'coil', 't'}
%                   Note that the order of mergeDims in the input
%                   determines their order in the mergedImage, i.e.
%                   I.merge({'coil', 't'}) is different from I.merge({'t',
%                   'coil'}). Also, the mergedDim is always the last in the
%                   new image; use permute to change that.
%
%   varargin        prop/val pairs to describe the new dimension
%                   including resolutions, ranges, dimLabels, units,
%                   samplingPoints, samplingWidths
%                   defaults:   samplingPoints = 1:nSamples
%                               dimLabels = [oldDimLabels(1), '_', ...
%                               '_', olDimLabels(end)]
%                               units = ''
% OUT
%
% EXAMPLE
%   merge
%
%   See also MrImage

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

mergedImage = this.copyobj();
newDimLabel = [];
if ~isempty(mergeDims)
    % create merged dimInfo first - this also give the new labels and sampling
    % points
    [mergedDimInfo, ~, newDimLabel] = this.dimInfo.merge(mergeDims, varargin);
    
    % only split if more than one mergeDim is actually given
    if numel(mergeDims) > 1
        % split into individual images
        split_array = this.split('splitDims', mergeDims);
        split_array = reshape(split_array, 1, []);
        
        % overwrite the sampling points to the newly created ones
        for n = 1:numel(split_array)
            split_array{n}.dimInfo = mergedDimInfo.select(newDimLabel, n);
        end
        
        % use combine to merge image along non-singleton dimension
        mergedImage = split_array{1}.combine(split_array);
    else % otherwise just change the dimLabel
        mergedImage.dimInfo = mergedDimInfo;
    end
end
end