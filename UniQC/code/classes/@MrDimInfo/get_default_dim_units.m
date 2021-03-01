function defaultDimUnit = get_default_dim_units(this, iDim)
% returns default dim label for specified dimensions (nifti-compatible)
% convention: [x y z t coil echo dL7 dL8 ...]
%
%   defaultDimUnit = get_default_dim_units(this,iDim)
%
% This is a method of class MrDimInfo.
%
% IN
%   iDim                index of dimension
%
% OUT
%   defaultDimUnit      string of default dimension label
%
% EXAMPLE
%   get_default_dim_units
%
%   See also MrDimInfo MrDimInfo.set_dims MrDimInfo.add_dims MrDimInfo.get_default_dim_labels

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2018-02-21
% Copyright (C) 2018 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

% output for one dim
if numel(iDim) == 1
    defaultUnits6D = {'mm', 'mm', 'mm', 's', 'nil', 'ms'};
    % to allow '' unit...which is a string, but empty :-)
    if iDim < 7 % use default units
        defaultDimUnit = defaultUnits6D{iDim};
    else
        defaultDimUnit = 'nil';
    end
    
else % loop over dims
    for n = 1:numel(iDim)
        defaultDimUnit{n} = this.get_default_dim_units(iDim(n));
    end
end
end