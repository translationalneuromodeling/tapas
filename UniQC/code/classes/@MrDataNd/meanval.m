function meanValue = meanval(this, varargin)
% Returns mean value of data matrix of MrImage, accepts selection parameters
%
%   Y = MrImage()
%   meanValue = ...
%       Y.meanval('ParameterName1', 'ParameterValue1', ...)
%
% This is a method of class MrImage.
% NOTE: This is a deviation from the nameanvalg convention to use typical matlab
% functions in their original way, since the creation of a mean image is
% more frequently used than the retrieval of an overall mean valuel, and
% hence called mean(Y)
%
% IN
%   varargin    parameterName/Value pairs for selection of volumes/slices
%
% OUT
%
% EXAMPLE
%   Y.mean(50, 'z', 1, 't', 3:100, ...,
%           'x', 55:75)
%
% EXAMPLE
%   mean(Y)
%
%   See also MrImage MrImage.mean

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

meanValue = mean(imgSelect.data(:));