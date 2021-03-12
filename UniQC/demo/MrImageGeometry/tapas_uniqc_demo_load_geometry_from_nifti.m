% Script demo_load_geometry_from_nifti
% load geometry from nifti
%
%  demo_load_geometry_from_nifti
%
%
%   See also

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2018-10-18
% Copyright (C) 2018 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3. 
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>. 

fileNifti = fullfile(tapas_uniqc_get_path('examples'), 'nifti', 'rest', 'meanfmri.nii'); 

dimInfo = MrDimInfo(fileNifti);
figure; stem(dimInfo.samplingPoints{1}); % origin at centre of block - note
% how none of the voxels is exactly at zero (because of the even number of
% samples)
affineTransformation = MrAffineTransformation(fileNifti); % scaling still has resolution
affineTransformationOrig = MrAffineTransformation(fileNifti);
%% by hand
ADimInfo = dimInfo.get_affine_matrix;
affineTransformation.update_from_affine_matrix(affineTransformation.affineMatrix/ADimInfo)

geometry = MrImageGeometry(dimInfo,affineTransformation);

disp(geometry.resolution_mm - affineTransformationOrig.scaling);
disp(geometry.shear - affineTransformationOrig.shear)
disp(geometry.rotation_deg - affineTransformationOrig.rotation_deg);
disp(geometry.offcenter_mm - affineTransformationOrig.offcenter_mm);

%% now in MrAffineTransformation
clear affineTransformation
affineTransformation = MrAffineTransformation(fileNifti, dimInfo);
geometry2 = MrImageGeometry(dimInfo, affineTransformation);
disp(geometry2.isequal(geometry)); % true

% try with affine matrix as well
affineTransformation2 = MrAffineTransformation(affineTransformationOrig.affineMatrix, dimInfo);
geometry3 = MrImageGeometry(affineTransformation2, dimInfo);
disp(geometry.isequal(geometry3)); % true

%% now directly in MrImageGeometry
geometry4 = MrImageGeometry(fileNifti);
disp(geometry.isequal(geometry4)); % true

%% illustrate world space operations
image = MrImage(fileNifti);
[dimInfo, affineTrafo] = ...
    image.geometry.perform_world_space_operation('shift', [0 20 3], image.dimInfo);
disp(dimInfo.isequal(image.dimInfo)); % true
disp(affineTrafo.isequal(image.affineTransformation)); % false
disp(['old offcenter_mm: ', num2str(image.affineTransformation.offcenter_mm)]);
disp(['new offcenter_mm: ', num2str(affineTrafo.offcenter_mm)]);