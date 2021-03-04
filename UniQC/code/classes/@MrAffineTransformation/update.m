function [this, argsUnused] = update(this, varargin)
% updates Geometry directly or computes given new FOV/resolution/nVoxels or whole affineTransformation
%
%   Y = MrImageGeometry()
%   Y.update('FOV_mm', FOV_mm, 'resolution_mm', resolution_mm, ...
%            'nVoxels', nVoxels, 'affineMatrix', affineMatrix, ...
%            <other geom propertyName/Value pairs>)
%
% This is a method of class MrImageGeometry.
%
% Two of the three values FOV/resolution/nVoxels have to be given to define
% the third. Alternatively, the 4x4 affine Matrix can be given to update
% the geometry.
% Other geom parameters are just re-written directly (offcenter,
% rotation...)
%
% IN
%
% OUT
%   this        updated MrImageGeometry
%   argsUnused  other ParamName/Value pairs that do not correspond to image
%               geometry
%
% EXAMPLE
%   update
%
%   See also MrImageGeometry

% Author:   Lars Kasper
% Created:  2016-01-30
% Copyright (C) 2016 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.


% ignore cases without input to update...
if nargin > 1
    
    defaults.affineMatrix =  [];
    defaults.scaling = this.scaling;
    defaults.offcenter_mm = [];
    defaults.rotation_deg = [];
    defaults.shear = this.shear;

    
    [argsGeom, argsUnused] = tapas_uniqc_propval(varargin, defaults);
    tapas_uniqc_strip_fields(argsGeom);
    
    updateAffine = ~isempty(affineMatrix);  
    
    % here, computations are necessary
    if updateAffine
        this.update_from_affine_transformation_matrix(affineMatrix);
    else
        this.scaling            = scaling;
        this.offcenter_mm       = offcenter_mm;
        this.rotation_deg       = rotation_deg;
        this.shear           = shear;
    end  
end
end