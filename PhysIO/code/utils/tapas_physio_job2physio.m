function physio = tapas_physio_job2physio(job)
% Converts job from SPM batch editor to physio-structure
%
%   physio = tapas_physio_job2physio(job)
%
% IN
%
% OUT
%   physio  physio input structure, as use by
%           tapas_physio_main_create_regressors
%
% EXAMPLE
%   physio = tapas_physio_job2physio(job)
%
%   See also tapas_physio_cfg_matlabbatch tapas_physio_main_create_regressors

% Author: Lars Kasper
% Created: 2015-01-05
% Copyright (C) 2015 TNU, Institute for Biomedical Engineering, University of Zurich and ETH Zurich.
%
% This file is part of the TAPAS PhysIO Toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.




physio                      = tapas_physio_new();

%% Use existing properties that are cfg_choices in job to overwrite
% properties of physio and set corresponding method

physio = tapas_physio_update_from_job(physio, job, ...
    {'preproc.cardiac.posthoc_cpulse_select', ...
    'preproc.cardiac.initial_cpulse_select', 'scan_timing.sync'}, ...
    {'preproc.cardiac.posthoc_cpulse_select', ...
    'preproc.cardiac.initial_cpulse_select', 'scan_timing.sync'}, ...
    true, ...
    'method');

%% Take over model substructs as is
modelArray =  ...
    {'movement', 'retroicor', 'rvt', 'hrv', ...
    'noise_rois', 'other'};

physio = tapas_physio_update_from_job(physio, job, ...
    strcat('model.', modelArray), strcat('model.', modelArray), ...
    true, 'include');

%% Convert yes => true (=1) and no => false (=0)
nModels = numel(modelArray);
for iModel = 1:nModels
    physio.model.(modelArray{iModel}).include = strcmpi(...
        physio.model.(modelArray{iModel}).include, 'yes');
end

%% Take over yes/no substructs as is, yes/no will become 'include' property
yesNoArray =  ...
    {'preproc.cardiac.filter'};

physio = tapas_physio_update_from_job(physio, job, ...
    yesNoArray, yesNoArray,  true, 'include');

%% Convert yes => true (=1) and no => false (=0)
nChoices = numel(yesNoArray);
for iChoice = 1:nChoices
    try
        eval(sprintf(['physio.%s.include = strcmpi(' ...
            'physio.%s.include, ''yes'');'], yesNoArray{iChoice}, ...
            yesNoArray{iChoice}));
    catch err
        tapas_physio_log(sprintf('No property %s defined in job (error: %s)', ...
            yesNoArray{iChoice}, err.message), [], 1);
    end
end

%% Use existing properties in job to overwrite properties of physio
physio = tapas_physio_update_from_job(physio, job, ...
    {'preproc.cardiac.modality', 'preproc.respiratory', 'scan_timing.sqpar', ...
    'log_files', 'verbose', 'save_dir',...
    'model.orthogonalise', 'model.censor_unreliable_recording_intervals', ...
    'model.output_multiple_regressors', ...
    'model.output_physio'}, ...
    {'preproc.cardiac.modality', 'preproc.respiratory', 'scan_timing.sqpar', ...
    'log_files', 'verbose', 'save_dir',...
    'model.orthogonalise', 'model.censor_unreliable_recording_intervals', ...
    'model.output_multiple_regressors', ...
    'model.output_physio'}, ...
    false);
