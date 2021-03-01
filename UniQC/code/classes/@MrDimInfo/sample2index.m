function arrayIndices = sample2index(this, samplingPoints, iDims)
% Returns voxel samplingPoints corresponding to label (coordinate) samplingPoints
%
%   Y = MrDimInfo()
%   arrayIndices = sample2index(this, samplingPoints)
%
% Inverse operation to get_samplingPoints
%
% This is a method of class MrDimInfo.
%
% IN
%   samplingPoints     [nVoxels, nDims] of voxel samplingPoints
%                      (one per row) in coordinate system given by dimInfo
%   iDims              if specified, samplingPoints are assumed to be subset
%                      of all dimensions only
%
% OUT
%   arrayIndices        [nVoxels, nDims] of absolute
%                       voxel samplingPoints within array
%
% EXAMPLE
%   sample2index
%
%   See also MrDimInfo MrDimInfo.get_samplingPoints

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2016-01-23
% Copyright (C) 2016 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

if nargin < 3
    iDims = 1:this.nDims;
end

nDimsSubset = numel(iDims);

isRowVector = nDimsSubset == 1 && size(samplingPoints,1) == 1;

if isRowVector
   samplingPoints = samplingPoints(:); % allows row/column vectors as input for 1-dim trafo
end

nVoxels = size(samplingPoints,1);

arrayIndices = zeros(nVoxels,nDimsSubset);

for v = 1:nVoxels
    % find voxel index with closest (euclidean) sampling point in array
    for d = 1:nDimsSubset  
        iDim = iDims(d);
        [~, arrayIndices(v,d)] = ...
            min(abs(this.samplingPoints{iDim} - samplingPoints(v,d)));
    end
end

if isRowVector % transform back to row vector for output
    arrayIndices = reshape(arrayIndices, 1, []);
end