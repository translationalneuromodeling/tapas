function [realignedImage, realignmentParameters] = realign(this, varargin)
% Realigns all 3D images in 4D data to each other using SPM's realign...
% estimate+rewrite functionality; interfaces all options of spm_realign
%
%   Y = MrImageSpm4D()
%   [realignedImage, realignmentParameters] = Y.realign(...
%       'spmParameterName1', spmParameterValue1, ...
%       ...
%       'spmParameterNameN', spmParameterValueN)
%
% This is a method of class MrImageSpm4D.
%
% IN
%   most SPM realign est/reslice parameters, enforcing congruency between 
%   est/reslice and ignoring file naming options:
%
%   quality         0..1, estimation quality, share of voxels included in estimation
%                   default: 0.9
%   separation      separation distance (mm) between evaluated image points in estimation
%                   default: 4
%   smoothingFwhm   FWHM (mm) of Gaussian smoothing kernel used for estimation
%                   default: 5
%   realignToMean   boolean; if true, 2-pass procedure, registering to mean
%                   default: true
%   interpolation   degree of b-spline interpolation for estimation and reslicing
%                   default: 7
%   wrapping        fold-over direction (phase encode)
%                   default: [0 0 0] % none
%   weighting       weighting image for estimation
%                   can be filename or MrImage
%                   default: '' % none
%   masking         mask incomplete timeseries?
%                   default: true
%
% OUT
%   realignmentParameters
%                   (from rp_*.txt)
%                   in mm and rad: [dx,dy,dz,pitch,roll,yaw]
%                                           (i.e., phi_x,phi_y,phi_z)
% EXAMPLE
%   Y = MrImageSpm4D()
%   [rY, realignmentParameters] = Y.realign('quality', 0.99, ...
%       'smoothingFwhm', 2);
%
%   [rY, realignmentParameters] = Y.realign('quality', 0.99, ...
%       'weighting', weightingMrImage);
%
% See also MrImage.realign spm_realign spm_cfg_realign

% Author:   Saskia Klein & Lars Kasper
% Created:  2014-07-08
% Copyright (C) 2014 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

% most SPM realign est/reslice parameters, enforcing congruency between 
% est/reslice and ignoring file naming options
% See also spm_realign or spm_cfg_realign

defaults.quality = 0.9;         % 0..1, estimation quality, share of voxels included in estimation
defaults.separation = 4;        % separation distance (mm) between evaluated image points in estimation
defaults.smoothingFwhm = 5;     % FWHM (mm) of Gaussian smoothing kernel used for estimation
defaults.realignToMean = 1;     % boolean; if true, 2-pass procedure, registering to mean
defaults.interpolation = 7;     % degree of b-spline interpolation for estimation and reslicing
defaults.wrapping = [0 0 0];    % fold-over direction (phase encode)
defaults.weighting = '';        % weighting image for estimation
defaults.masking = 1;           % mask incomplete timeseries?

args = tapas_uniqc_propval(varargin, defaults);

realignedImage = this.copyobj();

% save image file for processing as nii in SPM
realignedImage.save('fileName', realignedImage.get_filename('prefix', 'raw'));

% if input weighting is an MrImage, save it for execution as nifti, and
% give file name to matlabbatch
if isa(args.weighting, 'MrImage')
    fileWeighting = spm_file(realignedImage.get_filename('prefix', 'raw'), 'prefix', 'weighting_');
    args.weighting.save('fileName', fileWeighting);
    args.weighting = {fileWeighting};
end

% convert file name string into cell of string, if not already
if ~isempty(args.weighting)
    args.weighting = cellstr(args.weighting);
end

matlabbatch = realignedImage.get_matlabbatch('realign', args);

save(fullfile(realignedImage.parameters.save.path, 'matlabbatch.mat'), ...
    'matlabbatch');
spm_jobman('run', matlabbatch);

% clean up: move/delete processed spm files, load new data into matrix

realignmentParameters = realignedImage.finish_processing_step('realign', ...
    args.weighting);
end
