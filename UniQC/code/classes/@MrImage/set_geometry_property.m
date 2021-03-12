function outputImage = set_geometry_property(this, varargin)
% Allows the setting of specific values in the geometry.
%
%   Y = MrImage()
%   Y.set_geometry_property(inputs)
%
% This is a method of class MrImage.
%
% NOTE: This is a method of MrImage rather than MrImageGeometry, because
%       the latter is composed on the fly from affineTransformation and
%       dimInfo to integrate both information and sustain consistency.
%
% NOTE: The property is *set* as specified in the property value pair to a
%       given value. This is in contrast to the shift, rotate, shear and
%       zoom methods, where the value is added to the existing image
%       geomtry.
% IN
%       property/values pairs for
%           resolution_mm
%           offcenter_mm
%           rotation_deg
%           shear
%           origin_mm
%
% OUT
%
% EXAMPLE
%   set_geometry_property
%
%   See also MrImage

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2019-12-09
% Copyright (C) 2019 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

outputImage = this.copyobj();

% defaults are the values in this, i.e. nothing is changed
defaults.resolution_mm  = outputImage.geometry.resolution_mm;
defaults.offcenter_mm   = outputImage.geometry.offcenter_mm;
defaults.rotation_deg   = outputImage.geometry.rotation_deg;
defaults.shear          = outputImage.geometry.shear;

% get arguments
[args, ~] = tapas_uniqc_propval(varargin, defaults);
inputVarargin = tapas_uniqc_struct2propval(args);

% create temp MrImageGeometry with new parameters
tmpGeometry = MrImageGeometry([], inputVarargin{:});

% update the affine transformation, but take the dimInfo into account
outputImage.affineTransformation = MrAffineTransformation(...
    tmpGeometry.get_affine_matrix(), outputImage.dimInfo);

end


