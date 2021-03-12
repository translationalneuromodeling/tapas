function otherImage = permute(this, order)
% Permutes array dimensions (and corresponding)
%
%   Y = MrImage()
%   permutedImage = Y.permute(order)
%
% This is a method of class MrImage.
%
% IN
%   order   [1, nDims] vector or permutation of 1:nDims indicating order
%           of dimensions after permutation. 
%           Note: if only a subset of 1:nDims are given, other dimensions
%           are appended to be kept in right order
%           e.g. [2 4] will be appended to [2 4 1 3] for 4D data
% OUT
%   permutedImage
%           permuted image, with permuted dimInfo and geometry
%   
% EXAMPLE
%   permutedY = Y.permute([2 4]);
%   % permutes image dimensions 2 and 4 to first two, 
%
%   See also MrImage

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2016-04-06
% Copyright (C) 2016 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the  TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3. 
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

% compute numerical index, if dimLabels given
if ~isnumeric(order)
    order = this.dimInfo.get_dim_index(order);
end

% append unspecified dimensions in permutation
if this.dimInfo.nDims > numel(order)
    sfxOrder = setdiff(1:this.dimInfo.nDims, order);
    order = [order, sfxOrder];
end

otherImage = this.copyobj;
otherImage.data = permute(otherImage.data, order);
otherImage.dimInfo.permute(order);

otherImage.info{end+1,1} = sprintf('permute(this, [%s]);', sprintf('%d ', ...
    order));