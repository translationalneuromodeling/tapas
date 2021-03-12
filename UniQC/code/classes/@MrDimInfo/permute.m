function this = permute(this, order)
% Permutes dimension info given a specified dimension order
%
%   Y = MrDimInfo()
%   Y.permute(order)
%
% This is a method of class MrDimInfo.
%
% IN
%   order   [1, nDims] vector or permutation of 1:nDims indicating order
%           of dimensions after permutation. 
%           Note: if only a subset of 1:nDims are given, other dimensions
%           are appended to be kept in right order
%           e.g. [2 4] will be appended to [2 4 1 3] for 4D data
% OUT
%
% EXAMPLE
%   dimInfo.permute([2 4]);
%
%   See also MrDimInfo

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2016-04-06
% Copyright (C) 2016 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3. 
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.


if ~isequal(order, 1:this.nDims)
    % otherwise: already ordered
    
    if numel(order) < this.nDims
        sfxOrder = setdiff(1:this.nDims, order);
        order = [order, sfxOrder];
    else
        order = order(1:this.nDims);
    end
    
    this.dimLabels = this.dimLabels(order);
    this.units = this.units(order);
    this.samplingPoints = this.samplingPoints(order);
    this.samplingWidths = this.samplingWidths(order);
end