% Script demo_model_estimation_1st_level
% 1st level model specification and estimation
%
%  demo_model_estimation_1st_level
%
%
%   See also

% Author:   Saskia Bollmann & Lars Kasper
% Created:  2018-05-04
% Copyright (C) 2018 Institute for Biomedical Engineering
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
%% (0) User Inputs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% uses the output of demo_preprocessing; PLEASE MODIFY TO YOUR PATH
pathOutputDemoPreprocessing = fullfile(pwd, 'preprocessing', 'MrSeries_*');
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% (1) Load data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

S = MrSeries(pathOutputDemoPreprocessing);
% change directory to get a separate it from the preprocessing
S.parameters.save.path = strrep(S.parameters.save.path, 'preprocessing', 'model_estimation');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% (2) Make brain mask
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% add white and grey matter TPMs
S.additionalImages{end+1} = S.tissueProbabilityMaps{1} + S.tissueProbabilityMaps{2};
S.additionalImages{4}.name = 'brain_tpm';
S.additionalImages{4}.parameters.save.fileName = 'brain_tpm.nii';
S.additionalImages{4}.plot;
% set parameters for mask
S.parameters.compute_masks.nameInputImages = 'brain_tpm';
S.parameters.compute_masks.nameTargetGeometry = 'mean';
S.compute_masks;
% check mask
S.mean.plot('overlayImages', S.masks{2});
% close mask
S.masks{3} = S.masks{2}.imclose(strel('disk', 5));
% check again
S.mean.plot('overlayImages', S.masks{3});
% perfect - save new mask
S.masks{3}.save;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% (3) Specify Model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% timing in seconds
S.glm.timingUnits = 'secs';
% repetition time - check!
disp(['The specified TR is ', num2str(S.data.geometry.TR_s), 's.']);
S.glm.repetitionTime = S.data.geometry.TR_s;
% model derivatives
S.glm.hrfDerivatives = [1 1];

% add conditions
% specify block length
block_length = 20;
% specify condition onset
condition = 20:40:380;
% remove 5 first TRs
condition_onsets = condition - 5 * S.data.geometry.TR_s;

% add to glm
S.glm.conditions.names = {'tapping'};
S.glm.conditions.onsets = {condition_onsets};
% add durations
S.glm.conditions.durations = {block_length};
% add an explicit mask
S.glm.explicitMasking = S.masks{3}.get_filename;
% turn of inplicit masking threshold;
S.glm.maskingThreshold = -Inf;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% (4) Estimate Model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compute stat images
S.compute_stat_images;
% estimate
S.specify_and_estimate_1st_level;
% look at design matrix
S.glm.plot_design_matrix;