function medianValue = median(this, varargin)
% Returns median value of data matrix of MrImage by applying prctile(50)
%
%   Y = MrImage()
%   medianValue = ...
%       Y.median('ParameterName1', 'ParameterValue1', ...)
%
% This is a method of class MrImage.
%
% IN
%   varargin    parameterName/Value pairs for selection of volumes/slices
%
% OUT
%
% EXAMPLE
%   Y.median(50, 'z', 1, 'r', 3:100, ...,
%           'x', 55:75)
%
% EXAMPLE
%   median(Y)
%
%   See also MrImage MrImage.medianip

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


medianValue = this.prctile(50, varargin{:});