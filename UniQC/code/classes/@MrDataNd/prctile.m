function percentile = prctile(this, percentile, varargin)
% Computes given percentile in data (e.g. 50 for median)
%
%   Y = MrImage()
%   percentile = ...
%       Y.prctile(percentile, 'ParameterName1', 'ParameterValue1', ...)
%
% This is a method of class MrImage.
%
% IN
%   percentile  percent for percentile computation; default: 50 (median)
%   varargin    parameterName/Value pairs for selection of volumes/slices
%
% OUT
%
% EXAMPLE
%   Y.prctile(50, 'z', 1, 't', 3:100, ...,
%           'x', 55:75)
%
%   See also MrImage MrImage.hist

% Author:   Lars Kasper
% Created:  2015-08-23
% Copyright (C) 2015 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3. 
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.


if nargin < 2
    percentile = 50;
end

if nargin < 3
    imgSelect = this;
else
    imgSelect = this.select(varargin{:});
end

percentile = prctile(imgSelect.data(:), percentile);