function outputImage = shuffle(this, iDimArray, shuffledSamplingPointsIndexArray)
% Shuffles sampling points of specified dimensions, and dimInfo-description
% (?)
%
%   Y = MrDataNd()
%   Y.shuffle(inputs)
%
% This is a method of class MrDataNd.
%
% IN
%
% OUT
%
% EXAMPLE
%   shuffle({'y','z'}, {[nSamples:-1:1], [1:2:nSamples, 2:2:nSamples]}),
%
%   See also MrDataNd

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2019-10-10
% Copyright (C) 2019 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

outputImage = this.copyobj;

nDims = numel(iDimArray);
for iDim = 1:nDims
    idxDim = this.dimInfo.get_dim_index(iDimArray{iDim});
    dataSelectorString = repmat({':'}, 1, this.dimInfo.nDims);
    dataSelectorStringNew = dataSelectorString;
    dataSelectorStringNew{idxDim} = shuffledSamplingPointsIndexArray{iDim};
    outputImage.data(dataSelectorString{:}) = ...
        outputImage.data(dataSelectorStringNew{:});
    % TODO: do we adapt the dimInfo as well?
end