function samplingPointLabels = index2label(this, arrayIndices, iDims)
% Returns array of index labels for specified voxel indices
%
%   Y = MrDimInfo()
%   indexLabelArray = index2label(this, arrayIndices)
%
% For 3D data, this usually returns the voxel coordinates of the specified
% voxel indices, for 4D fMRI, it additionally outputs the acquisition onset
% of the specified onsets, and so forth, all in their respective units
%
% This is a method of class MrDimInfo.
%
% IN
%   arrayIndices        matrix [nVoxels, nDims] of index vectors (in rows)
%                       specifying the position of the voxels in the
%                       multi-dimensional array
%   iDims               if specified, arrayIndices are assumed to be subset
%                       of all dimensions only
%
% OUT
%   samplingPointLabels     cell(nVoxel,1)  of {1,nDims} sample label cells
%                           e.g. {'x = 13 mm', 'y = 20 mm', 'z = -130 mm',
%                           'volume = 30 s', 'coil = 33', 'echo = 17 ms'};
%
% EXAMPLE
%   index2label
%
%   See also MrDimInfo

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

isRowVector = nDimsSubset == 1 && size(arrayIndices,1) == 1;

if isRowVector
    arrayIndices = arrayIndices(:); % allows row/column vectors as input for 1-dim trafo
end

samplingPoints = this.index2sample(arrayIndices, iDims);

% Loop to print sampling points with units
nVoxels = size(arrayIndices,1);
samplingPointLabels = cell(nVoxels,1);
for v = 1:nVoxels
    samplingPointLabels{v} = cell(1,nDimsSubset);
    
    for d = 1:numel(iDims)
        samplingPointLabels{v}{d} = sprintf('%s = %.1f %s', ...
            this.dimLabels{iDims(d)}, samplingPoints(v,d), this.units{iDims(d)});
    end
end

if isRowVector
    samplingPointLabels = reshape(samplingPointLabels, 1, []);
end
