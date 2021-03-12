function iDim = get_singleton_dimensions(this)
% returns vector of indices of singleton dimensions (i.e. 1 or less elements)
%
%   Y = MrDimInfo()
%   Y.get_singleton_dimensions(inputs)
%
% This is a method of class MrDimInfo.
%
% IN
%
% OUT
%
% EXAMPLE
%   get_singleton_dimensions
%
%   See also MrDimInfo

% Author:   Lars Kasper
% Created:  2018-05-03
% Copyright (C) 2018 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3. 
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

iDim = find(cell2mat(cellfun(@(x) numel(x) <= 1, this.samplingPoints, ...
    'UniformOutput', false)));