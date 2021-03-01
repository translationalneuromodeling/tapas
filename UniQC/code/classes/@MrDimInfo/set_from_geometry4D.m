function this = set_from_geometry4D(this, geometry)
% initializes dimInfo from MrImageGeometry (nifti etc) with standard labels
%
%   Y = MrDimInfo()
%   Y.set_from_geometry4D(geometry)
%
% This is a method of class MrDimInfo.
%
% IN
%   geometry    MrImageGeometry (4D nifti...)
%
% OUT
%
% EXAMPLE
%   set_from_geometry4D
%
%   See also MrDimInfo

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2017-10-12
% Copyright (C) 2017 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.


this.set_from_affine_geometry(geometry, geometry.nVoxels, geometry.TR_s);