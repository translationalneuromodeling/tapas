function this = add_dims(this, iDim, varargin)
% Adds additional dimensions to existing dimInfo
%
%   Y = MrDimInfo()
%   Y.add_dims(iDims, varargin)
%
% This is a method of class MrDimInfo.
%
% IN
%   iDim        (vector of) dimension indices to be added (e.g. 3 for 3rd
%               dimension)
%                   or
%               (cell of) strings of dimension names
%   varargin    PropertyName/Value pairs to set parameters of the new
%               dimension
%
% OUT
%
% EXAMPLE
%   add_dims('coil', 'units', 'nil', 'samplingPoints', {1:8})
%
%   See also MrDimInfo set_dims

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2016-06-18
% Copyright (C) 2016 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.


% nothing to do here...
if isempty(iDim)
    return
end


%% To update a dimension (set_dims), it has to have a label first
% This can be given explicitly or using the default dimLabels of MrDimInfo

% later permutation at specified index, setup here
doPermute = isnumeric(iDim);
if doPermute
    iDimNumeric = iDim;
end

% direct adding of a labeled dimension, e.g., add_dims('coil', ...)
isStringiDimInput = ischar(iDim) || (iscell(iDim) && ischar(iDim{1}));
if isStringiDimInput
    additionalDimLabels = cellstr(iDim);
else
    % add_dims via their index, e.g. add_dims([4:6}, propName/Value...)
    if ~iscell(iDim)
        iDim = num2cell(iDim);
    end
    
    if isempty(iDim{1})
        % empty set of dimLabels to add, therefore return
        return
    else
        
        additionalDimLabels = cellfun(@(x) this.get_default_dim_labels(x), iDim, ...
            'UniformOutput', false);
    end
end

% empty defaults here will be overwritten by set_dims
nDimsOld = this.nDims;
nDimsAdditional = numel(additionalDimLabels);
this.dimLabels = [this.dimLabels additionalDimLabels];
this.samplingPoints(nDimsOld+(1:nDimsAdditional)) = {[]};
this.resolutions(nDimsOld+(1:nDimsAdditional)) = 1;
this.samplingWidths(nDimsOld+(1:nDimsAdditional)) = NaN;
this.units(nDimsOld+(1:nDimsAdditional)) = {''};
% set_dims also needed to add samplingPoints (e.g. via Nsamples/resolution)
if nargin > 2
    this.set_dims(additionalDimLabels, varargin{:});
end

% permute here to corresponding index
if doPermute
    iDim = iDimNumeric;
    nDimsNew = nDimsOld + nDimsAdditional;
    
    % current order of dimensions after appending the dimensions at the end
    dimOrderNew = [setdiff(1:nDimsNew, iDim), iDim];
    
    % desired permutation to have right dimensions at their place
    invDimOrderNew(dimOrderNew) = 1:length(dimOrderNew); % inverse permutation
    this.permute(invDimOrderNew);
end