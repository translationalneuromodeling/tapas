function this = load(this, fileName)
% Loads Geometry info from affine image header (.nii/.hdr/.img) or Philips
% (par/rec)
%
% NOTE: .mat-header files (for 4D niftis) are ignored, since the same voxel
%       position is assumed in each volume for MrImage
%
%   geom = MrImageGeometry()
%   geom.load(fileName)
%
% This is a method of class MrImageGeometry.
%
% IN
%
% OUT
%
% EXAMPLE
%   geom = MrImageGeometry()
%   geom.load('test.nii')
%   geom.load('test.hdr/img')
%   geom.load('test.par/rec')
%
%   See also MrImageGeometry MrImageGeometry.update

% Author:   Saskia Klein & Lars Kasper
% Created:  2014-07-15
% Copyright (C) 2014 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public Licence (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.


affineMatrix = [];
TR_s = [];
if exist(fileName, 'file')
    [fp, fn, ext] = fileparts(fileName);
    
    switch ext
        case {'.hdr', '.nii', '.img'}
            V = spm_vol(fileName);
            affineMatrix = V.mat;
            
            this.nVoxels = [V(1).dim numel(V)];
            
            if ~isempty(affineMatrix)
                this.update('affineMatrix', affineMatrix);
            end
            
            % some nifti formats supply timing information
            if isfield(V(1), 'private')
                if isstruct(V(1).private.timing)
                    this.TR_s = V(1).private.timing.tspace;
                end
            end
            
            
            
        case {'.par', '.rec'}
            this.load_par(fileName);
            
        otherwise
            warning('Only Philips (.par/.rec), nifti (.nii) and analyze (.hdr/.img) files are supported');
    end
else
    fprintf('Geometry data could not be loaded: file %s not found.\n', ...
        fileName);
end


end


