function otherImage = circshift(this, nShiftSamples, applicationDimensions)
% circularly shifts data matrix by nShiftSamples for all given dimensions
%
%   Y = MrImage()
%   Y.circshift(nShiftSamples, iDim)
%
% This is a method of class MrImage.
%
% IN
%   nShiftSamples       [1, nShiftDims] number of samples to shift
%                       (negative to shift left, positive to shift right
%   applicationDimensions [1, nShiftsDims]
%                       indices of all dimensions along which image shall
%                       be shifted
% OUT
%
% EXAMPLE
%   circshift([20, 10], [2 1]);
%
%   See also MrImage  built-in/circshift

% Author:   Lars Kasper
% Created:  2016-04-05
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
    applicationDimensions = 1;
end

otherImage = this.copyobj;

for iDim = reshape(applicationDimensions, 1, [])
    otherImage.data = circshift(this.data, nShiftSamples(iDim), iDim);

    otherImage.dimInfo.samplingPoints{iDim} = ...
        circshift(this.dimInfo.samplingPoints{iDim}, ...
        nShiftSamples(iDim));
end