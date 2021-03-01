function this = remove_dims(this, iDim)
% Removes information of dimension(s) specified (by names or index)
%
%   Y = MrDimInfo()
%   Y.remove_dims(inputs)
%
% This is a method of class MrDimInfo.
% Use cases are the selection of single values from one dimensions to loose
% all information on the specifie ddimension
%
% IN
%   iDim        (vector of) dimension index or cell array of names 
%               to be changed (e.g. 3 for 3rd
%               dimension) or (cell of) strings of dimension names
%               default: remove singleton-dimensions (with only 1 or 0 sampling
%               points)
%               []  = remove all singleton dimensions
% OUT
%
% EXAMPLE
%   remove_dims
%
%   See also MrDimInfo

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2016-04-03
% Copyright (C) 2016 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3. 
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

if nargin < 2 || isempty(iDim)
    iDim = this.get_singleton_dimensions();
end

isStringiDimInput = ~isempty(iDim) && (ischar(iDim) || (iscell(iDim) && ischar(iDim{1})));
if isStringiDimInput
    dimLabel = iDim;
    iDim = this.get_dim_index(dimLabel);
elseif iscell(iDim)
    iDim = cell2mat(iDim);
end

% Leave dim-info at least one-dimensional, otherwise errors with other
% access methods, e.g. min/max in MrImage
if isequal(iDim, 1:this.nDims)
    iDim = setdiff(iDim,1);
end

this.dimLabels(iDim)        = [];
this.units(iDim)            = [];
this.samplingPoints(iDim)   = [];
this.samplingWidths(iDim)   = [];