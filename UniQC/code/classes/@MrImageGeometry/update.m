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
    
    defaults.affineMatrix = [];
    defaults.FOV_mm = [];
    defaults.nVoxels = [];
    defaults.resolution_mm = [];
    defaults.shear = [];
    
    defaults.offcenter_mm = [];
    defaults.rotation_deg = [];
    defaults.TR_s = [];
   
    [argsGeom, argsUnused] = tapas_uniqc_propval(varargin, defaults);
    tapas_uniqc_strip_fields(argsGeom);
    
    updateOffcenter = ~isempty(offcenter_mm);
    updateRotation = ~isempty(rotation_deg);
    updateTR = ~isempty(TR_s);
    updateShear = ~isempty(shear);
    
    
    hasRes = ~isempty(this.resolution_mm);
    hasFOV = ~isempty(this.FOV_mm);
    
    hasUpdateValueNvoxels = ~isempty(nVoxels);
    
    hasUpdateValueResOrFOV = ~isempty(resolution_mm) || ~isempty(FOV_mm);
    
    
    updateNvoxels = ~isempty(resolution_mm) && ~isempty(FOV_mm);
    
    updateResOrFOV = ~updateNvoxels && hasUpdateValueResOrFOV || ...
        (hasUpdateValueNvoxels && (hasRes || hasFOV));
    
    updateAffine = ~isempty(affineMatrix);
    
    
    % direct overwriting, do it!
    if updateOffcenter
        this.offcenter_mm = offcenter_mm;
    end
    
    if updateRotation
        this.rotation_deg = rotation_deg;
    end
    
    if updateTR
        this.TR_s = TR_s;
    end
    
    
    if updateShear
        this.shear = shear;
    end
    % here, computations are necessary
    if updateAffine
        this.update_from_affine_matrix(affineMatrix);
        
        if ~hasUpdateValueNvoxels
            nVoxels = this.nVoxels;
        end
        
        nVoxels((end+1):4) = 1;
        this.FOV_mm = nVoxels(1:3).*this.resolution_mm;
        this.nVoxels = nVoxels;
        
    else % explicit setting of nVoxels, resolution_mm or FOV_mm
        
        if updateNvoxels
            nVoxels = this.nVoxels;
            nVoxels(1:3) = ceil(FOV_mm./resolution_mm);
            % per default, 4D-nVoxels is output
            nVoxels((end+1):4) = 1;
            
        elseif updateResOrFOV
            
            %% nVoxels given, sth else must be updated
            if ~hasUpdateValueNvoxels
                nVoxels = this.nVoxels;
            end
            
            %% If no input FOV/Res, take existing one in geometry,
            % with precedence to resolution
            if ~hasUpdateValueResOrFOV
                if hasRes
                    resolution_mm = this.resolution_mm;
                elseif hasFOV
                    FOV_mm = this.FOV_mm;
                else
                    error('tapas:uniqc:MrImageGeometry:UnspecifiedFOVResolution', ...
                        ['No FOV_mm and resolution_mm given in %s. - ' ...
                        'Don''t know what else to do update with nVoxels.' ], ...
                        inputname(1));
                end
            end
            
            %% update FOV or Res depending on which one is given, using nVoxels
            
            updateFOV = ~isempty(resolution_mm);
            updateRes = ~isempty(FOV_mm);
            
            
            % per default, 4D-nVoxels is output
            nVoxels((end+1):4) = 1;
            
            
            if updateFOV
                FOV_mm = nVoxels(1:3).*resolution_mm;
            elseif updateRes
                resolution_mm = FOV_mm./nVoxels(1:3);
            end
            
            
        else
            error('tapas:uniqc:MrImageGeometry:UnknownUpdateDependency', ...
                'unknown update dependency for geometry %s', inputname(1));
        end
        
        this.FOV_mm = FOV_mm;
        this.resolution_mm = resolution_mm;
        this.nVoxels = nVoxels;
        
    end
end
end