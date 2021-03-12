% Script make_example_data
% Creates example data for uniQC-demos based on the ME-data in Heunis et al.
%
%  make_example_data
%
% NOTE: This script has to be run only once, in order to prepare some 
%       example files for all uniqc-demos from a publicly available imaging
%       dataset.
%
% INSTRUCTIONS:
% 1. Please download the following data 
%   (we only need subject sub-001 and sub-021) separately at
%    https://dataverse.nl/dataverse/rt-me-fmri.
%
%   from 
%   Heunis, Stephan, 2020, "rt-me-fMRI: A task and resting state dataset for
%   real-time, multi-echo fMRI methods development and validation",
%   https://doi.org/10.34894/R1TNL8, DataverseNL, V1. 
%
% 2. Unzip the downloaded subject folders
%   
% 3. Update the paths in section "User Input" below, i.e.
%       a) where the downloaded, unzipped data resides (parent folder of sub-001)
%       b) where the generated uniqc example data files shall be stored
%          (for future use)
%
% 4. Run this script
%

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2021-02-26
% Copyright (C) 2021 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, which is released
% under the terms of the GNU General Public License (GPL), version 3. 
% You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.
 
clear; close all; clc; 
 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% User Inputs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% file path to the Heunis-example data (we only need sub-001)
% PLEASE MODIFY TO YOUR PATH
sourceDataPath = 'path/to/downloaded/me_sh_data';

% where do want to store the example data for uniQC - this is also the path
% you will be asked for when running the examples
% PLEASE MODIFY TO YOUR PATH
exampleDataPath = '/path/to/save/uniQC-examples';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load and plot source data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
m = MrImage(fullfile(sourceDataPath, 'sub-001', 'func', ...
    'sub-001_task-fingerTapping_echo-2_bold.nii'));
m.plot();

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Create small EPI data sets
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% nifti/rest/meanfmri.nii - just the mean of a times series (3D image)
meanfmri = m.mean();
meanfmri.parameters.save.path = fullfile(exampleDataPath, 'nifti', 'rest');
meanfmri.parameters.save.fileName = 'meanfmri.nii';
meanfmri.save();

% nifti/rest/fmri_short.nii - a short section of the fMRI time series to
% speed up processing in the demos
fmri_short = m.select('t', 1:15);
fmri_short.parameters.save.path = fullfile(exampleDataPath, 'nifti', 'rest');
fmri_short.parameters.save.fileName = 'fmri_short.nii';
fmri_short.save();

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Save anatomy
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
s = MrImage(fullfile(sourceDataPath, 'sub-001', 'anat', 'sub-001_T1w.nii'));
s.parameters.save.path = fullfile(exampleDataPath, 'nifti', 'rest');
s.parameters.save.fileName = 'struct.nii';
s.save();

% uniQC creates an extra mat file to keep track of any dimensions not saved
% in the nifti file, we delete it here to simulate the use case that we
% usually start only with a nifti
delete(fullfile(exampleDataPath, 'nifti', 'rest', '*.mat'));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Create 5D file
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% nifti/5D/y_5d_deformation_field.nii
% this could for example be on of the deformation fields estimated during
% segmentation (the large samplingDistance makes this a quick and dirty
% version)
segmentInput = meanfmri.remove_dims.copyobj();
segmentInput.parameters.save.keepCreatedFiles = 'processed';
segmentInput.parameters.save.fileName = '5D.nii';
segmentInput.parameters.save.path = fullfile(exampleDataPath, 'nifti', '5D');

[~, ~, deformationFields] = segmentInput.remove_dims.segment(...
    'samplingDistance', 20, 'tissueTypes', {'GM'});
movefile(deformationFields{1}.get_filename, ...
    fullfile(exampleDataPath, 'nifti', '5D', 'y_5d_deformation_field.nii'));
delete(fullfile(exampleDataPath, 'nifti', '5D', 'bias*'));
delete(fullfile(exampleDataPath, 'nifti', '5D', 'tissue*'));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Create split file
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% now load the complete multi-echo data set
meFileNames = dir(fullfile(sourceDataPath, 'sub-001', 'func', 'sub-001_task-fingerTapping_echo-*_bold.nii'));
for nFiles = 1:numel(meFileNames)
    % load files into temporary array
    meTmp{nFiles} = MrImage(fullfile(sourceDataPath, 'sub-001', 'func',...
        meFileNames(nFiles).name));
    % get echo time from json data
    metaInfo = jsondecode(fileread(fullfile(sourceDataPath, 'sub-001', 'func', ...
        regexprep(meFileNames(nFiles).name, '.nii', '.json'))));
    meTmp{nFiles}.dimInfo.add_dims('TE', ...
        'samplingPoints', metaInfo.EchoTime*1000, 'units', 'ms');
end

% combine into 5D image
me = meTmp{1}.combine(meTmp);

% save using the uniQC convention
% nift/split/subset/fmri_***.nii
meSplit = me.select('t', [1, 7, 8], 'TE', [1, 2]);
meSplit.parameters.save.path = fullfile(exampleDataPath, 'nifti', 'split', 'subset');
meSplit.parameters.save.fileName = 'fmri.nii';
meSplit.save('splitDims', {'t', 'TE'});
delete(fullfile(exampleDataPath,'nifti', 'split', 'subset', '*.mat'));

% nifti/data_multi_echo/multi_echo.nii
me.parameters.save.path = fullfile(exampleDataPath, 'nifti', 'data_multi_echo');
me.parameters.save.fileName = 'multi_echo.nii';
me.save();
delete(fullfile(exampleDataPath, 'nifti', 'data_multi_echo', '*.mat'));

% save without uniQC conventions
% /nifti/split/split_residual_images/Res.nii'
for n = 1:5
    currentM = me.select('t', n, 'TE', 1).remove_dims();
    currentM.parameters.save.path = fullfile(exampleDataPath, 'nifti', 'split', 'split_residual_images');
    currentM.parameters.save.fileName = ['Res_', num2str(n, '%04.0f'), '.nii'];
    currentM.save();
end
delete(fullfile(exampleDataPath, 'nifti', 'split', 'split_residual_images', '*.mat'));

% save with uniQC dimInfo
% /nifti/split/with_dim_info/MrImage.nii'
meSplitWith = me.select('t', [1, 7, 8]);
meSplitWith.parameters.save.path = fullfile(exampleDataPath, 'nifti', 'split', 'with_dim_info');
meSplitWith.parameters.save.fileName = 'MrImage.nii';
meSplitWith.save();

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Create data for first level
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% /nifti/data_first_level/
% structural image
s.parameters.save.path = fullfile(exampleDataPath, 'nifti', 'data_first_level');
s.parameters.save.fileName = 'anatomy.nii';
s.save();

% third echo as functional image
fl = me.select('TE', 3).remove_dims();
fl.parameters.save.path = fullfile(exampleDataPath, 'nifti', 'data_first_level');
fl.parameters.save.fileName = 'single_echo.nii';
fl.save();
delete(fullfile(exampleDataPath, 'nifti', 'data_first_level', '*.mat'));
