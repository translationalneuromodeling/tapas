function this = set_from_geometry4D(this, geometry)
% Updates affine transformation from read-in MrImageGeometry
%
%   Y = MrAffineTransformation()
%   Y.set_from_geometry4D(geometry)
%
% This is a method of class MrAffineTransformation.
%
% IN
%   geometry        MrImageGeometry
% OUT
%
% EXAMPLE
%   set_from_geometry4D
%
%   See also MrAffineTransformation

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


% MrAffineMatrix is always in nifti coordinate system
geometryNifti = geometry.copyobj();

this.shear = geometryNifti.shear;
this.rotation_deg = geometryNifti.rotation_deg;
this.scaling = geometryNifti.resolution_mm;
this.offcenter_mm = geometryNifti.offcenter_mm;

