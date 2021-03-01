function this = permute(this, order)
% Permutes image geometry
%
%   Y = MrImageGeometry()
%   Y.permute(order)
%
% This is a method of class MrImageGeometry.
%
% IN
%   order   [1, 3] vector or permutation of 1:3 indicating order
%           of dimensions after permutation. 
%           Note: if only a subset of 1:3 is given, other dimensions
%           are appended to be kept in right order
%           e.g. [2 3] will be appended to [2 3 1] for 4D data
%
% OUT
%
% EXAMPLE
%   geometry.permute([2 3])
%
%   See also MrImageGeometry

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


if numel(order) < 4
    sfxOrder = setdiff(1:4, order);
    order = [order, sfxOrder];
else
    order = order(1:4);
end

order3 = order(1:3);

this.FOV_mm         = this.FOV_mm(order3);
this.nVoxels        = this.nVoxels(order);
this.resolution_mm   = this.resolution_mm(order3);
this.offcenter_mm   = this.offcenter_mm(order3);
this.rotation_deg   = this.rotation_deg(order3);
this.shear       = this.shear(order3);

% TODO: figure out how slice orientation is affected by permutation!
% Assumption here: slice orientation just specifies which of the previous
% dimensions will become the slice dimension...it's a relative measure
% though...
% this.slice_orientation = order(3); 
