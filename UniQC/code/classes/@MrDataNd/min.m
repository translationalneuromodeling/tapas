function [minValue, minPosition] = min(this, varargin)
% Returns minimum value of data matrix of MrImage, accepts selection parameters
%
%   Y = MrImage()
%   [minValue, minPosition] = ...
%       Y.min('ParameterName1', 'ParameterValue1', ...)
%
% This is a method of class MrImage.
%
% IN
%   varargin    parameterName/Value pairs for selection of volumes/slices
%
% OUT
%   minValue    minimum value in whole data array
%   minPosition [1,nDims] vector of voxel indices (over dimensions) of 
%               location of minimum
%
% EXAMPLE
%   Y.min(50, 'z', 1, 'r', 3:100, ...,
%           'x', 55:75)
%
% EXAMPLE
%   min(Y)
%
%   See also MrImage MrImage.minip

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2014-11-25
% Copyright (C) 2014 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public Licence (GPL), version 3. 
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.


if nargin < 2
    imgSelect = this;
else
    imgSelect = this.select(varargin{:});
end


[minValue, minIndex] = min(imgSelect.data(:));

% convert index to subscript array of arbitrary dimension
minPosition = cell(1,imgSelect.dimInfo.nDims);
[minPosition{:}] = ind2sub(imgSelect.dimInfo.nSamples, minIndex);
minPosition = [minPosition{:}];