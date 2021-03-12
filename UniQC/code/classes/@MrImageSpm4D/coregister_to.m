function [outputImage, affineCoregistrationGeometry, outputOtherImages] = coregister_to(this, ...
    stationaryImage, varargin)
% Coregister this MrImage to another given MrImage
% NOTE: Also does reslicing of image
%
%   Y = MrImageSpm4D()
%   [outputImage, affineCoregistrationGeometry, outputOtherImages] = 
%   Y.coregister_to(stationaryImage, ...
%   'applyTransformation', 'data', 'trafoParameters', 'rigid',
%   'otherImage', Z)
%
% This is a method of class MrImageSpm4D.
%
% IN
%       stationaryImage  MrImage that serves as "stationary" or reference image
%                        to which this image is coregistered to
%
%  optional parameter name/value pairs:
%       applyTransformation
%                   'geometry'      MrImageGeometry is updated,
%                                   MrImage.data remains untouched
%                   'data'          MrImage.data is resliced to new
%                                   geometry
%                                   NOTE: An existing
%                                   transformation in MrImageGeometry will
%                                   also be applied to MrImage, combined
%                                   with the calculated one for
%                                   coregistration
%
%                   'none'          transformation matrix is
%                                   computed, but not applied to geometry of data of this
%                                   image
%       trafoParameters             'translation', 'rigid', 'affine', 'rigidscaled' or
%                                   [1,1-12] vector of starting parameters
%                                   for transformation estimation
%                                   number of elements decides whether
%                                   translation only (1-3)
%                                   rigid (4-6)
%                                   rigid and scaling (7-9)
%                                   affine (10-12)
%                                   transformation is performed
%                                   default: 'rigid' (as in SPM)
% SPM input parameters:
%          separation           optimisation sampling steps (mm)
%                               default: [4 2]
%          objectiveFunction    cost function string:
%                               'mi'  - Mutual Information
%                               'nmi' - Normalised Mutual Information
%                               'ecc' - Entropy Correlation Coefficient
%                               'ncc' - Normalised Cross Correlation
%                               default: 'nmi'
%          tolerances           tolerances for accuracy of each param
%                               default: [0.02 0.02 0.02 0.001 0.001 0.001]
%          histSmoothingFwhm    smoothing to apply to 256x256 joint histogram
%                               default: [7 7]
%          otherImages          cell(nImages,1) of other images (either
%                               file names or MrImages) that should undergo
%                               the same trafo as this images due to coreg
%                               default: {}
%           doPlot              set to true for graphical output and PS file creation
%                               in SPM graphics window 
%                               default: false
%
%
% OUT
%       affineCoregistrationGeometry    MrImageGeometry holding mapping from
%                                       stationary to transformed image
%       outputOtherImages               coregistered other images, same
%                                       transformation applied to as to "this"
%                                       image
%
% EXAMPLE
%   Y = MrImageSpm4D();
%   stationaryImage = MrImageSpm4D();
%
%   co-registers Y to stationaryImage, i.e. changes geometry of Y
%   cY = Y.coregister_to(stationaryImage);
%
%   See also MrImage spm_coreg

% Author:   Saskia Klein & Lars Kasper
% Created:  2014-07-24
% Copyright (C) 2014 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public Licence (GPL), version 3.
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

% SPM parameters
spmDefaults.doPlot = false; % set to true for graphical output and PS file creation
spmDefaults.otherImages = {};
spmDefaults.objectiveFunction = 'nmi';
spmDefaults.separation = [4 2 1];
spmDefaults.tolerances = [0.02 0.02 0.02 0.001 0.001 0.001 0.01 0.01 0.01 0.001 0.001 0.001];
spmDefaults.histSmoothingFwhm = [7 7];
spmDefaults.trafoParameters = 'rigid';
% will be forwarded to get_matlabbatch to update the batch
[spmParameters, unusedVarargin] = tapas_uniqc_propval(varargin, spmDefaults);

% method parameters
defaults.applyTransformation = 'data';
defaults.otherImages = {};
args = tapas_uniqc_propval(unusedVarargin, defaults);
tapas_uniqc_strip_fields(args);

% create output image
outputImage = this.copyobj();

%% SPM needs nifti files, but otherImages could be MrImage and have to be saved to
% disk; file name creation happens here
otherImages = spmParameters.otherImages;
spmParameters.otherImages = {}; % will be populated with file names below

% single other images given instead of cell array?
if ~iscell(otherImages)
    otherImages = {otherImages};
end

nOtherImages = numel(otherImages);
outputOtherImages = cell(nOtherImages, 1);
for iImage = 1:nOtherImages
    if ~isa(otherImages{iImage}, 'MrImage')
        % file name given
        % load file as MrImage for later internal reslicing
        outputOtherImages{iImage} = MrImage(otherImages{iImage});
    else
        % MrImage object given; copy object
        outputOtherImages{iImage} = otherImages{iImage}.copyobj();
    end
    % add filename to spmParameters
    spmParameters.otherImages{iImage} = otherImages{iImage}.get_filename();
end

%% standard settings for transformation estimation starting points define type
% of estimated coregistration transformation
if ~isnumeric(spmParameters.trafoParameters)
    % otherwise can be taken as input values directly
    switch lower(spmParameters.trafoParameters)
        case 'affine'
            spmParameters.trafoParameters = [0 0 0 0 0 0 1 1 1 0 0 0];
        case 'rigid'
            spmParameters.trafoParameters = [0 0 0 0 0 0];
        case 'rigidscaled'
            spmParameters.trafoParameters = [0 0 0 0 0 0 1 1 1];
        case 'translation'
            spmParameters.trafoParameters = [0 0 0];
    end
end


%% save raw and stationary image data as nifti
% set filenames
spmParameters.stationaryImage = cellstr(...
    fullfile(outputImage.parameters.save.path, 'rawStationary.nii'));

% save raw files
outputImage.save('fileName', outputImage.get_filename('prefix', 'raw'));
stationaryImage.save('fileName', spmParameters.stationaryImage{1});

%% matlabbatch
% get matlabbatch
matlabbatch = outputImage.get_matlabbatch('coregister_to', ...
    spmParameters);
% save matlabbatch
save(fullfile(outputImage.parameters.save.path, 'matlabbatch.mat'), ...
    'matlabbatch');

% NOTE: outputImage job is not actually run to enable a clean separation of
% coregistration and re-writing of the object
% spm_jobman('run', matlabbatch);
% NOTE: The following lines are copied and modified from spm_run_coreg to
% enable a separation between computation and application of coregistration
% parameters

job = matlabbatch{1}.spm.spatial.coreg.estimate;

%% Coregistration
% Compute coregistration transformation
x  = spm_coreg(char(job.ref), char(job.source), job.eoptions);

% Apply coregistration, if specified, but leave raw image untouched!

% header of stationary image:
% MatF voxel -> world
% header of transformed image:
% MatV voxel -> world
%
% transformation in spm_coreg:
% worldF -> worldF

%  mapping from voxels in G to voxels in F is attained by:
%           i.e. from reference to source:
%               G = reference
%               F = source
%
%         VF.mat\spm_matrix(x(:)')*VG.mat
% =       inv(VF.mat) * spm_matrix(x) * VG.mat
% A\B = inv(A) * B

% get affine coregistration matrix
affineCoregistrationMatrix = tapas_uniqc_spm_matrix(x);
affineCoregistrationGeometry = MrAffineTransformation(affineCoregistrationMatrix);

%% update geometry/data if necessary
doUpdateAffineTransformation = ismember(applyTransformation, {'data', 'geometry'});
% update geometry
if doUpdateAffineTransformation
    % output image
    outputImage.affineTransformation.apply_inverse_transformation(affineCoregistrationGeometry);
    % other images
    for iImage = 1:nOtherImages
        outputOtherImages{iImage}.affineTransformation.apply_inverse_transformation(affineCoregistrationGeometry);
    end
end

% reslice image
doResliceImages = strcmpi(applyTransformation, 'data');
if doResliceImages
    % reslice output image
    % keep save parameters for later, s.t.
    % coregister-finish-processing-step can clean up properly later
    parametersSave = outputImage.parameters.save;
    outputImage.parameters.save.keepCreatedFiles = 1;
    % reslice image to given geometry
    outputImage = outputImage.reslice(stationaryImage.geometry);
    outputImage.parameters.save = parametersSave;
    
    % reslice other images
    % finish_processing_step of reslice will take care of created images
    for iImage = 1:nOtherImages
        % reslice image to given geometry
        outputOtherImages{iImage} = outputOtherImages{iImage}.reslice(stationaryImage.geometry);
    end
else
    % save processed images to disk
    % s.t. reload in finish_processing_step has correct new header
    outputImage.save();
end

%% clean up: move/delete processed spm files, load updated data and geom into
% outputImage
fnOutputSpm = {};
outputImage.finish_processing_step('coregister_to', ...
    spmParameters.stationaryImage{1}, ...
    fnOutputSpm);