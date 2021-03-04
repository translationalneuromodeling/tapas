%% Demonstrates basic and high dimensional functionality of (affine) image coregistration
% wrapping spm_coreg
clear;
close all;
clc;

%% load data

pathData        = tapas_uniqc_get_path('examples');

fileFunctional      = fullfile(pathData, 'nifti', 'rest', 'fmri_short.nii');
fileFunctionalMean  = fullfile(pathData, 'nifti', 'rest', 'meanfmri.nii');
fileStructural      = fullfile(pathData, 'nifti', 'rest', 'struct.nii');

% stationary image is the mean functional
func = MrImage(fileFunctionalMean);
func.parameters.save.fileName = 'funct.nii';
% moving image is the structural
anat = MrImage(fileStructural);
anat.parameters.save.fileName = 'struct.nii';

%% 0) Introducing the data and plot (voxel space)
func.plot;
anat.plot('z', 1:10:anat.dimInfo.nSamples('z'));

% spm check registration
anat.plot('plotType', 'spmi', 'overlayImages', func);


%% I. Cases directly mimicking SPM's coreg behavior for different parameter settings
% nomenclature: 
% this          altered image (source, moving)
% stationary    reference image (coregister_to(stationary))
% other         other images to which estimated coregistration is applied after
%               estimation of this -> stationary trafo
% 
%% 1a) this: 3D; stationary: 3D; but only update geometry
%  -> trivial case, directly mimicking SPM's coregister and passing to
%  MrImageSpm4D
[cZG, rigidCoregistrationTrafoG] = anat.coregister_to(func, ...
    'applyTransformation', 'geometry');

disp(rigidCoregistrationTrafoG);

% looks the same as before (voxel-space plot)
cZG.plot('z', 1:10:anat.dimInfo.nSamples('z')); 

% but looks different in checkreg... (respects world space)
% Note: A good way to check the alignment of images is the contour plot
% option offered in the SPM checkreg view. Simply right-click on the mean
% functional image and select 'contour -> Display onto -> all'. The corpus
% callosum and the ventricles are an area with good contrast to check the
% registration.
cZG.plot('plotType', 'spmi', 'overlayImages', {func, anat});

%% 1b) Coregister with reslicing of data
[cZD, rigidCoregistrationTrafoD] = anat.coregister_to(func, ...
'applyTransformation', 'data', 'separation', 4, 'tolerances', [1 1 1 1 1 1]);
disp(rigidCoregistrationTrafoD);

% compare to the mean functional image - both are now in the same voxel
% space
cZD.plot();
cZD.plot('overlayImages', func, 'overlayMode', 'edge');
%% 1c) Coregister with other images
otherImages = {anat.log(), anat.^-1};
otherImages{1}.plot(); otherImages{2}.plot()

[cMO, rigidCoregistrationTrafoO, otherImagesO] = anat.coregister_to(func, ...
    'applyTransformation', 'data', 'otherImages', otherImages);

func.plot();
cMO.plot();
otherImagesO{1}.plot;
otherImagesO{2}.plot;

%% 1d) Estimate affine coregistration for comparison
[cZAffine, affineTrafo] = anat.coregister_to(func, ...
    'applyTransformation', 'geometry', 'trafoParameters', 'affine');
% rigid transformation
disp(rigidCoregistrationTrafoG);
% affine transformation
disp(affineTrafo); % somewhat different

%% 1e) Estimate using Normalised Cross Correlation as objective function
[outImageNCC, rigidCoregTrafoNCC] = anat.coregister_to(func, ...
'applyTransformation', 'geometry', 'objectiveFunction', 'ncc');
disp(rigidCoregTrafoNCC);

%% 1f) Estimate using smaller separation
[outImageSeparation, rigidCoregTrafoSeparation] = anat.coregister_to(func, ...
'applyTransformation', 'geometry', 'separation', [4 2 1 0.5]);
disp(rigidCoregTrafoSeparation); % doesn't change much

%% 1g) Estimate using higher tolerances
% tolerances are given for the upt to 12 affine transformation parameters,
% namely translation, rotation, scaling and shear
% here, we only use the first 6 for a rigid body estimation
[outImageTolerances, rigidCoregTrafoTolerances] = anat.coregister_to(func, ...
'applyTransformation', 'geometry', 'tolerances', [0.1 0.1 0.1 0.01 0.01 0.01]);
disp(rigidCoregTrafoTolerances); % not too bad

%% 1h) Estimate with more histogram smoothing
[outImageHistSmoothing, rigidCoregTrafohistSmoothing] = anat.coregister_to(func, ...
'applyTransformation', 'geometry', 'histSmoothingFwhm', [14 14]);
disp(rigidCoregTrafohistSmoothing); % pretty similar as well