% Script demo_image_geometry
% Exemplifies creation and usage of MrImageGeometry from nifti files,
% dimInfo and affineTrafo and par/re files
%
%  demo_image_geometry
%
%
%   See also MrImageGeometry

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2017-10-30
% Copyright (C) 2017 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

clear;
close all;
clc;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Create from nifti
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dataPath = tapas_uniqc_get_path('data');
niftiFile4D = fullfile(dataPath, 'nifti', 'rest', 'fmri_short.nii');
geom = MrImageGeometry(niftiFile4D);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Create from dimInfo and affineTransformation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dimInfo = MrDimInfo(niftiFile4D);
affineTransformation = MrAffineTransformation(niftiFile4D);

% this will lead to a wrong geometry, because both dimInfo and
% affineTransformation now contain the resolution and a translation
geom2Wrong = MrImageGeometry(dimInfo, affineTransformation);
disp(geom.isequal(geom2Wrong));

% there are several ways to obtain the correct geometry
% simply pass on the dimInfo that is used in the combination
affineTransformation = MrAffineTransformation(niftiFile4D, dimInfo);
geom2Correct = MrImageGeometry(dimInfo, affineTransformation);
disp(geom.isequal(geom2Correct));

% or perform the internal operation to take into account the dimInfo
affineTransformation = MrAffineTransformation(niftiFile4D);
affineTransformation.update_from_affine_matrix(...
    affineTransformation.get_affine_matrix()/dimInfo.get_affine_matrix()); 
geom2Alternative = MrImageGeometry(dimInfo, affineTransformation);
disp(geom.isequal(geom2Alternative));

% dimInfo only
% note how rotation and shear are lost and the offcentre_mm changes, because
% these are stored in affineTransformation
geom4 = MrImageGeometry(dimInfo);

% affineTransformation only
% note how FOV, resolution and offcentre change, because these are stored
% in the dimInfo
geom5 = MrImageGeometry(affineTransformation);