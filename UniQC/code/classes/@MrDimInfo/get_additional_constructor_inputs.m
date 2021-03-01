function defaults = get_additional_constructor_inputs(this)
% Returns additional (non-property) inputs that can be set in constructor
% of MrDimInfo, e.g., 'firstSamplingPoint', along with their defaults
%
%   Y = MrDimInfo()
%   Y.get_additional_constructor_inputs()
%
% This is a method of class MrDimInfo.
%
% IN
%
% OUT
%
% EXAMPLE
%   get_additional_constructor_inputs
%
%   See also MrDimInfo

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2019-11-08
% Copyright (C) 2019 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

defaults.arrayIndex = [];
defaults.samplingPoint = [];
defaults.firstSamplingPoint = [];
defaults.lastSamplingPoint = [];
defaults.originIndex = [];