function originIndex = get_origin(this, iDims, voxelOrVolumeSpace)
% returns (also fractional) index of samplingPoint with value [0 0 ... 0]
% for nifti counting, i.e. the first sample is [0 0 0 ...], whereas matlab
% starts counting a [1 1 1 ...]
%
%   Y = MrDimInfo()
%   originIndex = Y.get_origin(iDims)
%
% This is a method of class MrDimInfo.
%
% IN
%   iDims   [1,nDims] vector of dimension indices where origin shall be reported
%   voxelOrVolumeSpace
%           'voxel' or 'volume'
%           returned index of origin can be defined via
%           'volume': for matlab-related
%           index-counting within the volume (i.e. first voxel is
%           [1,1,...,1]) or
%           'voxel': for SPM/Nifti-realted index-counting within the
%           traditional voxel space (i.e, first voxel is [0, 0, ..., 0]);
%           default: 'voxel'
%
% OUT
%
% EXAMPLE
%   get_origin
%
%   See also MrDimInfo
%
% Author:   Saskia Bollmann & Lars Kasper
% Created:  2018-12-12
% Copyright (C) 2018 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.
if nargin < 2
    iDims = 1:this.nDims;
end

if nargin < 3
    voxelOrVolumeSpace = 'voxel';
end

originIndex = [];
nDims = numel(iDims);

% indices determined for counting from 1...nSamples
for d = 1:nDims
    iDim = iDims(d);
    if this.nDims >= iDim && ~isempty(this.samplingPoints{iDim})
        [~,originIndex(d)] = min(abs(this.samplingPoints{iDim}));
        fractionalIndex = this.index2sample(originIndex(d), iDim)./this.resolutions(iDim);
        originIndex(d) = originIndex(d) - fractionalIndex;
    end
end

% adjust voxels for voxel space
switch lower(voxelOrVolumeSpace)
    case 'volume'
        % all good
    case 'voxel' % voxel count starts at 0
        originIndex = originIndex - 1;
end

end