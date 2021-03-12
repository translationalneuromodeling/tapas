%
% Example analysis for snr assessment and preprocessing steps

% Author:   Saskia Klein & Lars Kasper
% Created:  2014-11-18
% Copyright (C) 2014 Institute for Biomedical Engineering
%                    University of Zurich and ETH Zurich
%
% This file is part of the TAPAS UniQC Toolbox, 
% which is released under the terms of the GNU General Public Licence (GPL), 
% version 3. You can redistribute it and/or modify it under the terms of 
% the GPL (either version 3 or, at your option, any later version).
% For further details, see the file COPYING or
%  <http://www.gnu.org/licenses/>.

clear;
close all;
clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% #MOD# The following parameters can be altered to analyze different image 
% time series
% default: funct_short (fMRI Philips 3T)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

pathExamples        = tapas_uniqc_get_path('examples');
pathData            = fullfile(pathExamples, 'nifti', 'rest');

fileFunctional      = fullfile(pathData, 'fmri_short.nii');
fileStructural      = fullfile(pathData, 'struct.nii');

dirResults          = ['results' filesep];



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Load data into time series
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

S = MrSeries(fileFunctional);
S.parameters.save.path = tapas_uniqc_prefix_files(S.parameters.save.path, ...
    dirResults);
S.anatomy.load(fileStructural, 'updateProperties', 'none');



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Compute statistical images (mean, snr, sd, etc.)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

S.compute_stat_images();
S.snr.plot('displayRange', [0 500]);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Compute tissue probability maps of anatomical image
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

S.parameters.compute_tissue_probability_maps.nameInputImage = 'anatomy';
S.compute_tissue_probability_maps();



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Coregister anatomy to mean functional and take tissue probability maps ...
%  with it
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

S.parameters.coregister.nameStationaryImage = 'mean';
S.parameters.coregister.nameTransformedImage = 'anatomy';
S.parameters.coregister.nameEquallyTransformedImages = 'tissueProbabilityMap';

S.coregister();



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Compute masks from co-registered tissue probability maps via thresholding
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

S.parameters.compute_masks.nameInputImages = 'tissueProbabilityMap';
S.parameters.compute_masks.nameTargetGeometry = 'mean';
S.parameters.compute_masks.threshold = 0.5;
S.parameters.compute_masks.keepExistingMasks = false;

S.compute_masks();



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Extract region of interest data for masks from time series data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

S.parameters.analyze_rois.nameInputImages = {'mean', 'sd', 'snr', ...
    'coeffVar', 'diffLastFirst'};
S.parameters.analyze_rois.nameInputMasks = '.*mask';
S.parameters.analyze_rois.keepCreatedRois = false;
S.analyze_rois();
S.snr.plot_rois('selectedRois', 1, 'dataGrouping', 'perVolume'); ylim([0 400]);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Do some fancy preprocessing to the time series to see how SNR increases
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

S.realign();
S.compute_stat_images();
S.snr.plot('displayRange', [0 500]);

% maybe necessary if geometry changed too much through realignment
% S.coregister();
% S.compute_masks();
S.analyze_rois();
S.snr.plot_rois('selectedRois', 1, 'dataGrouping', 'perVolume'); ylim([0 400]);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Do some fancy preprocessing to the time series to see how SNR increases
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
S.parameters.smooth.fwhmMillimeters
S.smooth();
S.compute_stat_images();
S.snr.plot('displayRange', [0 500]);
S.analyze_rois();
S.snr.plot_rois('selectedRois', 1, 'dataGrouping', 'perVolume'); ylim([0 400]);
