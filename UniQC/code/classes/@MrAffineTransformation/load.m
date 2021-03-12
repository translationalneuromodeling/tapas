function this = load(this, fileName)
% Loads Geometry info from affine image header (.nii/.hdr/.img) or Philips
% (par/rec) or recon6-ImageData (.mat, Geometry-object)
%
% NOTE: .mat-header files (for 4D niftis) are ignored, since the same voxel
%       position is assumed in each volume for MrImage
%
%   geom = MrAffineTransformation()
%   geom = geom.load(fileName)
%
% This is a method of class MrAffineTransformation.
%
% IN
%
%
% EXAMPLE
%   geom = MrAffineTransformation()
%   geom.load('test.nii')
%   geom.load('test.hdr/img')
%   geom.load('test.par/rec')
%
%   See also MrAffineTransformation MrAffineTransformation.update

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


if exist(fileName, 'file')
    [~, ~, ext] = fileparts(fileName);
    
    switch ext
        case {'.hdr', '.nii', '.img'}
            V = spm_vol(fileName);
            affineMatrix = V.mat;
            
            if ~isempty(affineMatrix)
                this.update_from_affine_matrix(affineMatrix);
            end
            
        case {'.par', '.rec'}
            this = this.load_par(fileName);
        case {'.mat'} % recon 6
            this = this.load_recon6_mat(fileName);
        otherwise
            warning('Only Philips (.par/.rec), nifti (.nii) and analyze (.hdr/.img) files are supported');
    end
else
    fprintf('Geometry data could not be loaded: file %s not found.\n', ...
        fileName);
end


end


