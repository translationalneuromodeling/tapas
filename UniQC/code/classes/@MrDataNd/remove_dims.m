function outputImage = remove_dims(this, iDim)
% removes dimension(s) specified (by names or index)
%
%   Y = MrDataNd()
%   Y.remove_dims(iDim)
%
% This is a method of class MrDataNd.
%
% Use cases are the selection of single values from one dimensions to lose
% all information on the specified dimension
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
%   See also MrDataNd

% Author:   Saskia Bollmann & Lars Kasper
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

outputImage = this.copyobj;

if nargin < 2 || isempty(iDim)
    iDim = outputImage.dimInfo.get_singleton_dimensions();
end

isStringiDimInput = ~isempty(iDim) && (ischar(iDim) || (iscell(iDim) && ischar(iDim{1})));
if isStringiDimInput
    dimLabel = iDim;
    iDim = outputImage.dimInfo.get_dim_index(dimLabel);
elseif iscell(iDim)
    iDim = cell2mat(iDim);
end

% Leave dim-info at least one-dimensional, otherwise errors with other
% access methods, e.g. min/max in MrImage
if isequal(iDim, 1:outputImage.dimInfo.nDims)
    iDim = setdiff(iDim,1);
end

outputImage.data = squeeze(outputImage.data); % TODO: this is more a collapse than a removal of a dimension...
outputImage.dimInfo.remove_dims(iDim);

end