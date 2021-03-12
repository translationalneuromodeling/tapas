function this = append(this, otherImage)
% appends other MrImage (of same 3D geometry) via 4th dim to end of MrImage
%
%   Y = MrImage()
%   Y.append(otherImage)
%  
%   OR
%   
%   Y.append(fileName);
%
%   OR
%   Y.append(dataMatrix);
%
% This is a method of class MrImage.
%
% IN
%
% OUT
%
% EXAMPLE
%   Y = MrImage();
%   otherImage = MrImage();
%
%   Y.append(otherImage);
%   Y.append('single_subj_T1.nii');
%
%   data = rand(128,128,33);
%   Y.append(data)
%
%   See also MrImage

% Author:   Saskia Klein & Lars Kasper
% Created:  2014-07-20
% Copyright (C) 2014 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public Licence (GPL), version 3. 
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.


% other image given as fileName 
if ischar(otherImage) 
    fileName = otherImage;
    otherImage = MrImage(fileName);
end

% other image given as data matrix
if isnumeric(otherImage) 
    dataMatrix = otherImage;
    otherImage = MrImage(dataMatrix);
    
    % copy geometry from this image, but adapt voxel dimensions
    otherImage.geometry = this.geometry.copyobj;
    otherImage.geometry.nVoxels(1:ndims(dataMatrix)) = size(dataMatrix);
    otherImage.geometry.nVoxels((ndims(dataMatrix)+1):end) = 1;
end

otherGeometry = otherImage.geometry.copyobj;
nVoxelsOther = otherGeometry.nVoxels;

 % for other nVolumes, 
 switch ndims(otherImage)
     case 3
         otherGeometry.nVoxels(4) = this.geometry.nVoxels(4);
     case 2
         otherGeometry.nVoxels(3:4) = this.geometry.nVoxels(3:4);
 end
 
[diffGeometry, isEqual, isEqualGeom] = this.geometry.diffobj(otherGeometry);

if ~isEqualGeom
    fprintf('Warning: Geometries do not match. Assuming first geometry for appending: \n');
end

switch ndims(otherImage)
    case 2
        this.data(:,:,end+1:end+nVoxelsOther(3),:) = ...
            otherImage.data;
        this.update_geometry_dim_info('nVoxels', ...
            this.geometry.nVoxels + [0 0 nVoxelsOther(3) 0]);
    case {3,4}
        this.data(:,:,:, end+1:end+nVoxelsOther(4)) = ...
            otherImage.data;
        this.update_geometry_dim_info('nVoxels', ...
            this.geometry.nVoxels + [0 0 0 nVoxelsOther(4)]);
end