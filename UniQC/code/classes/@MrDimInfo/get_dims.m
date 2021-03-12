function dimInfo = get_dims(this, iDim)
% Returns reduced dimInfo object just with specified dimensions
%
% This can be used e.g. for simple query of all dimInfo of one (or more)
% dimensions
%
%   dimInfo = MrDimInfo()
%   dimInfo.get_dims(iDim)
%
% This is a method of class MrDimInfo.
%
% IN
%   iDim        (vector of) dimension indices to be added (e.g. 3 for 3rd
%               dimension) 
%                   or 
%               (cell of) strings of dimension names
% OUT
%
% EXAMPLE
%   dimInfoX = dimInfo.get_dims('x')
%   dimInfoXY = dimInfo.get_dims({'x', 'y'})
%   % also concatenation is possible:
%   nSamplesXY = dimInfo.get_dims('x').nSamples * dimInfo.get_dims('y').nSamples
%
%   See also MrDimInfo

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


dimInfo = this.copyobj;

if nargin > 1 % selection specified!
    % strip all non-specified dimensions
    
    iRemoveDims = setdiff(1:this.nDims, this.get_dim_index(iDim));
    % for bookkeeping of permute, we cannot use absolute numerical indices and take labels instead
    dimLabels = this.dimLabels(this.get_dim_index(iDim));
    if ~isempty(iRemoveDims) % don't remove singleton dims accidentally!
        dimInfo.remove_dims(iRemoveDims);
    end    
    
    % permute to retain order for requested iDim, e.g. [3,2]
    if numel(iDim) > 1
        iDimOrder = dimInfo.get_dim_index(dimLabels);
        dimInfo.permute(iDimOrder);
    end
end