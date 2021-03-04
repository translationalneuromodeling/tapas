% Script demo_save
% Shows how to save (image) data to one or multiple files (split!)
%
%  demo_save
%
%
%   See also

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2016-09-22
% Copyright (C) 2016 Institute for Biomedical Engineering
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
%% 1. Load 4D Time series
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
dirSave             = 'results_demo_save';
pathExamples        = tapas_uniqc_get_path('examples');
pathData            = fullfile(pathExamples, 'nifti', 'rest');
pathData2           = fullfile(pathExamples, 'nifti', '5D');
fileFunctional      = fullfile(pathData, 'fmri_short.nii');

% 4D example
Y = MrImage(fileFunctional);

% 5D example
dimInfo2.dimLabels = {'x','y','z', 't', 'dr'};
dimInfo2.units = {'mm','mm','mm','t','mm'};
fileDeformationField = fullfile(pathData2, ...
    'y_5d_deformation_field.nii');
Y2 = MrImage(fileDeformationField, 'dimLabels',  {'x','y','z', 't', 'dr'}, ...
    'units', {'mm','mm','mm','t','mm'});


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 2. Save 3D, one file per volume
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

splitDims = {'t'}; % or: 4
fileName = fullfile(dirSave , 'fmri3D.nii');

[dimInfoSplit, sfxSplit, selectionSplit] = Y.dimInfo.split(splitDims);

% saves to results_demo_save/fmri3D_t0001...fmri3D_t0015.nii
Y.save('fileName', fileName, 'splitDims', splitDims);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 4. Save 2D, one file per deformation field direction and slice
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

splitDims = {'dr', 'z'};
fileName = fullfile(dirSave , 'deformed2D.nii');

% saves to results_demo_save/fmri3D_t0001...fmri3D_t0015.nii
Y2.save('fileName', fileName, 'splitDims', splitDims);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 5. Save with different bit depth, depending on image type and size
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% a) Create an 3D image with a large number of voxel (>=220x220x120)
Image3D = MrImage(rand(220, 220, 120));
% save image
Image3D.save;
out3D = dir(Image3D.get_filename);
disp(['File size in MB: ', num2str(out3D.bytes/1014/1024)]);
% compare precision between original and saved image
Image3DLoad = MrImage(Image3D.get_filename);
% smallest difference in samples (depends on the randomly generated
% numbers)
disp(['original struct: ', num2str(min(diff(sort(unique(Image3D.data(:))))), '%.40e'), ...
    ', saved struct: ', num2str(min(diff(sort(unique(Image3DLoad.data(:))))), '%.40e')]);

% b) Create a 4D image with the same number of voxel
Image4D = MrImage(rand(110, 110, 60, 8));
% save image
Image4D.save;
out4D = dir(Image4D.get_filename);
% has now half the file size
disp(['File size in MB: ', num2str(out4D.bytes/1014/1024)]);
% compare precision between original and saved image
Image4DLoad = MrImage(Image4D.get_filename);
% smallest difference in samples (depends on the randomly generated
% numbers)
disp(['original struct: ', num2str(min(diff(sort(unique(Image4D.data(:))))), '%.40e'), ...
    ', saved struct: ', num2str(min(diff(sort(unique(Image4DLoad.data(:))))), '%.40e')]);

% c) Create a 4D image with a large number of voxel (more than 2GiB in
% float, i.e. more than 2*1024*1024*1024*8/32 bit)
ImageLarge4D = MrImage(rand(1024, 1024, 1024/32, 2*8));
% save image
ImageLarge4D.save;
outLarge4D = dir(ImageLarge4D.get_filename);
% has now half the file size
disp(['File size in MB: ', num2str(outLarge4D.bytes/1014/1024)]);
% compare precision between original and saved image
ImageLarge4DLoad = MrImage(ImageLarge4D.get_filename);
% smallest difference in samples (depends on the randomly generated
% numbers) (will take some time to compute)
disp(['original struct: ', num2str(min(diff(sort(unique(ImageLarge4D.data(:))))), '%.40e'), ...
    ', saved struct: ', num2str(min(diff(sort(unique(ImageLarge4DLoad.data(:))))), '%.40e')]);
% file content has been transformed to integer, make sure data us scaled
% appropriately!
% use Shepp-Logan phantom to illustrate
data4D = repmat(phantom(1024), [1, 1, 1024/32, 2*8]);
ImageLarge4DScaled = MrImage(data4D * 1e4 + rand(1024, 1024, 1024/32, 2*8)*5e3);
ImageLarge4DScaled.plot;
% save image
ImageLarge4DScaled.save;
outLarge4DScaled = dir(ImageLarge4DScaled.get_filename);
% has now half the file size
disp(['File size in MB: ', num2str(outLarge4DScaled.bytes/1014/1024)]);
% compare precision between original and saved image
ImageLarge4DScaledLoad = MrImage(ImageLarge4DScaled.get_filename);
% smallest difference in samples (depends on the randomly generated
% numbers) (will take some time to compute)
disp(['original struct: ', num2str(min(diff(sort(unique(ImageLarge4DScaled.data(:))))), '%.40e'), ...
    ', saved struct: ', num2str(min(diff(sort(unique(ImageLarge4DScaledLoad.data(:))))), '%.40e')]);
ImageLarge4DScaled.plot('colorBar', 'on');
ImageLarge4DScaledLoad.plot('colorBar', 'on');
plot(ImageLarge4DScaled-ImageLarge4DScaledLoad, 'colorBar', 'on');
% note: noise structure changed due to transformation to integer values,
% but main feature of the image were retained