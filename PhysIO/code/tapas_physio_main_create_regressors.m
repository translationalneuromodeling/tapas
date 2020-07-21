function [physio, R, ons_secs] = tapas_physio_main_create_regressors(varargin)
% Main Toolbox Function for preprocessing & modelling from physio-structure
%
% [physio_out, R, ons_secs] = tapas_physio_main_create_regressors(physio)
%
%   OR
%
% [physio_out, R, ons_secs] = tapas_physio_main_create_regressors(...
%    log_files, scan_timing, preproc, model, verbose, save_dir);
%
% NOTE: All inputs in physio-structure have to be specified previous to
%       running this function.
%
% IN
%   physio      physio-structure, See also tapas_physio_new
%               OR
%               sub-structures of physio-structure, i.e.
%               log_files, sqpar, preproc, model, verbose, save_dir
%
% OUT
%   physio_out  modified physio-structure, contains read or computed values
%               in ons_secs, model.R and verbose.fig_handles
%   R           [nScans, nRegressors] multiple_regressors-matrix (i.e.
%               confound regressors for GLM)
%   ons_secs    updated "onsets in seconds" structure, containing raw and
%               filtered physiological time series and detected pulses,
%               also as vectors cropped to scan acquisition window
%
% REFERENCES
%
% RETROICOR     regressor creation based on Glover et al. 2000, MRM 44 and
%               Josephs et al. 1997, ISMRM 5, p. 1682
%               default model order based on Harvey et al. 2008, JMRI 28
% RVT           (Respiratory volume per time) Birn et al. 2008, NI 40
% HRV           (Heart-rate  variability) regressor creation based on
%               Chang et al2009, NI 44
%
% See also tapas_physio_new

% Author: Lars Kasper
% Created: 2011-08-01
% Copyright (C) 2011-2019 TNU, Institute for Biomedical Engineering, 
%               University of Zurich and ETH Zurich.
%
% This file is part of the TAPAS PhysIO Toolbox, which is released under
% the terms of the GNU General Public Licence (GPL), version 3. You can
% redistribute it and/or modify it under the terms of the GPL (either
% version 3 or, at your option, any later version). For further details,
% see the file COPYING or <http://www.gnu.org/licenses/>.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 0. Set Default parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% include subfolders of code to path as well
pathThis = fileparts(mfilename('fullpath'));
addpath(genpath(pathThis)); 

% These parameters could become toolbox inputs...
minConstantIntervalAlertSeconds     = 0.2;

if ~nargin
    error('Please specify a PhysIO-object as input to this function. See tapas_physio_new');
end

if nargin == 1 % assuming sole PhysIO-object as input
    physio      = varargin{1}; % first argument of function
else % assemble physio-structure
    physio = tapas_physio_new();
    physio.log_files    = varargin{1};
    physio.scan_timing  = varargin{2};
    physio.preproc      = varargin{3};
    physio.model        = varargin{4};
    physio.verbose      = varargin{5};
    physio.save_dir     = varargin{6};
end

% fill up empty parameters
physio = tapas_physio_fill_empty_parameters(physio);

% replace cellstrings
physio = tapas_physio_cell2char(physio);

% prepend absolute directories - save_dir
physio = tapas_physio_prepend_absolute_paths(physio);

% set sub-structures for readability; NOTE: copy by value, physio-structure
% not updated!
ons_secs    = physio.ons_secs;
save_dir    = physio.save_dir;
log_files   = physio.log_files;
preproc     = physio.preproc;
scan_timing = physio.scan_timing;
model       = physio.model;
verbose     = physio.verbose;

hasPhaseLogfile = strcmpi(log_files.vendor, 'CustomPhase');
doesNeedPhyslogFiles = model.retroicor.include || model.rvt.include || model.hrv.include;
hasPhyslogFiles = ~isempty(log_files.cardiac) || ~isempty(log_files.respiration);

if ~hasPhaseLogfile
    
    % read and preprocess logfiles only, if model-based physiological noise correction is needed
    if doesNeedPhyslogFiles
        
        
        if ~hasPhyslogFiles
            verbose = tapas_physio_log(['No physlog files specified, but models relying on ' ...
                'physiological recordings selected. I will skip those.'], ...
                verbose, 1);
            sqpar = scan_timing.sqpar;
        else
            
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %% 1. Read in vendor-specific physiological log-files
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            [ons_secs.c, ons_secs.r, ons_secs.t, ons_secs.cpulse, ons_secs.acq_codes, ...
                verbose] = tapas_physio_read_physlogfiles(...
                log_files, preproc.cardiac.modality, verbose, scan_timing.sqpar);
            
            % also: normalize cardiac/respiratory data, if wanted
            doNormalize = true;
            
            % Normalize and pad time series after read-In
            ons_secs = tapas_physio_preprocess_phys_timeseries(ons_secs, ...
                scan_timing.sqpar, doNormalize);
            
            
            hasCardiacData = ~isempty(ons_secs.c);
            hasRespData = ~isempty(ons_secs.r);
            
            
            verbose = tapas_physio_plot_raw_physdata(ons_secs, verbose);
            
            
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %% 2. Create scan timing nominally or from logfile
            % nominal:  using entered sequence parameters (nSlices, nScans etc)
            % Philips:  via gradient time-course or existing acq_codes in logfile
            % GE:       nominal
            % Siemens:  from tics (Release VD/E), from .resp/.ecg files (Release VB)
            % Biopac:   using triggers from Digital input (mat file)
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            [ons_secs, VOLLOCS, LOCS, verbose] = tapas_physio_create_scan_timing(...
                log_files, scan_timing, ons_secs, verbose);
            minConstantIntervalAlertSamples = ceil(minConstantIntervalAlertSeconds/ons_secs.dt);
            
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %% 3. Extract and preprocess physiological data, crop to scan aquisition
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            if hasCardiacData
                % preproc.cardiac.modality = 'OXY'; % 'ECG' or 'OXY' (for pulse oximetry)
                %% initial pulse select via load from logfile or autocorrelation with 1
                %% cardiac pulse
                
                [ons_secs.c, verbose] = tapas_physio_filter_cardiac(...
                    ons_secs.t, ons_secs.c, preproc.cardiac.filter, verbose);
                
                switch preproc.cardiac.initial_cpulse_select.method
                    case {'load_from_logfile', ''}
                        % do nothing
                    otherwise
                        % run one of the various cardiac pulse detection algorithms
                        [ons_secs.cpulse, verbose] = ...
                            tapas_physio_get_cardiac_pulses(ons_secs.t, ons_secs.c, ...
                            preproc.cardiac.initial_cpulse_select, ...
                            preproc.cardiac.modality, verbose);
                end
                
                
                %% post-hoc: hand pick additional cardiac pulses or load from previous
                %% time
                switch preproc.cardiac.posthoc_cpulse_select.method
                    case {'manual'}
                        % additional manual fill-in of more missed pulses
                        [ons_secs, outliersHigh, outliersLow, verbose] = ...
                            tapas_physio_correct_cardiac_pulses_manually(ons_secs, ...
                            preproc.cardiac.posthoc_cpulse_select, verbose);
                    case {'load'}
                        hasPosthocLogFile = exist(preproc.cardiac.posthoc_cpulse_select.file, 'file') || ...
                            exist([preproc.cardiac.posthoc_cpulse_select.file '.mat'], 'file');
                        
                        if hasPosthocLogFile % load or set selection to manual, if no file exists
                            osload = load(preproc.cardiac.posthoc_cpulse_select.file, 'ons_secs');
                            ons_secs = osload.ons_secs;
                        else
                            [ons_secs, outliersHigh, outliersLow, verbose] = ...
                                tapas_physio_correct_cardiac_pulses_manually(ons_secs,...
                                preproc.cardiac.posthoc_cpulse_select, verbose);
                        end
                    case {'off', ''}
                end
                
                % label constant samples as unreliable (clipping/detachment)
                [ons_secs.c_is_reliable, ~, verbose] = tapas_physio_detect_constants(ons_secs.c, ...
                    minConstantIntervalAlertSamples, [], verbose);
                ons_secs.c_is_reliable = 1 - ons_secs.c_is_reliable;
            end
            
            if hasRespData
                % filter respiratory signal
                ons_secs.fr = tapas_physio_filter_respiratory(ons_secs.r, ...
                    ons_secs.dt, doNormalize);
                
                % label constant samples as unreliable (clipping/detachment)
                [ons_secs.r_is_reliable, ~, verbose]  = tapas_physio_detect_constants(ons_secs.fr, ...
                    minConstantIntervalAlertSamples, [], verbose);
                ons_secs.r_is_reliable = 1 - ons_secs.r_is_reliable;
            end
            
            [ons_secs, scan_timing.sqpar, verbose] = tapas_physio_crop_scanphysevents_to_acq_window(...
                ons_secs, scan_timing.sqpar, verbose);
            sqpar = scan_timing.sqpar;
            
            
            if verbose.level >= 2
                verbose.fig_handles(end+1) = ...
                    tapas_physio_plot_cropped_phys_to_acqwindow(ons_secs, sqpar);
            end
            
            [verbose, ons_secs.c_outliers_low, ons_secs.c_outliers_high, ...
                ons_secs.r_hist] = ...
                tapas_physio_plot_raw_physdata_diagnostics(ons_secs.cpulse, ...
                ons_secs.r, preproc.cardiac.posthoc_cpulse_select, verbose, ...
                ons_secs.t, ons_secs.c);
        end
    else % does NOT NeedPhyslogFiles
        sqpar = scan_timing.sqpar;
    end % doesNeedPhyslogFiles
    
else
    
    % Phase data saved in log-file already
    % Read logged phases into object directly
    load(log_files.cardiac, 'c_phase_probe_regressors')
    load(log_files.respiratory, 'r_phase_probe_regressors');
    ons_secs.r_sample_phase = r_phase_probe_regressors;
    
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 4. Create physiological noise model regressors for GLM for all specified
%     slices
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

onset_slices = reshape(sqpar.onset_slice, 1, []);
nOnsetSlices = numel(onset_slices);

if nOnsetSlices < 1
    error('Please specify an onset slice.');
end

for onset_slice = onset_slices
    
    model.R = [];
    model.R_column_names = {};
    
    %% 4.1. Slice specific parameter adaptation
    
    sqpar.onset_slice = onset_slice;
    
    if hasPhaseLogfile
        % explicit down-sampling of pre-existing phases
        % for Field Probe + Physiological Noise Paper
        ons_secs.c_sample_phase = ...
            c_phase_probe_regressors((140+sqpar.onset_slice):(sqpar.Nslices):end);
        ons_secs.r_sample_phase = ...
            r_phase_probe_regressors((sqpar.onset_slice):(sqpar.Nslices):end);
    else
        % otherwise reset, since phases will be estimated from raw
        % c(ardiac) and r(espiratory) time courses
        ons_secs.c_sample_phase = [];
        ons_secs.r_sample_phase = [];
    end
    
    %% Physiological measures
    
    if hasPhyslogFiles
        
        %% 4.2. Create RETROICOR regressors (Fourier expansion of cardiac/respiratory phase)
        
        if model.retroicor.include
            [cardiac_sess, respire_sess, mult_sess, ons_secs, ...
                model.retroicor.order, verbose] = ...
                tapas_physio_create_retroicor_regressors(ons_secs, sqpar, ...
                model.retroicor.order, verbose);
            
            if model.censor_unreliable_recording_intervals
                [ons_secs, cardiac_sess, respire_sess, mult_sess, verbose] = ...
                    tapas_physio_censor_unreliable_regressor_parts_retroicor(...
                    ons_secs, sqpar, cardiac_sess, respire_sess, mult_sess, verbose);
            end
            
            [model.R, model.R_column_names] = append_regressors(model.R, model.R_column_names, ...
                cardiac_sess, 'RETROICOR (cardiac)');
            [model.R, model.R_column_names] = append_regressors(model.R, model.R_column_names, ...
                respire_sess, 'RETROICOR (respiratory)');
            [model.R, model.R_column_names] = append_regressors(model.R, model.R_column_names, ...
                mult_sess, 'RETROICOR (multiplicative)');
            
        end
        
        
        %% 4.3. Create a heart-rate variability regressor using the cardiac response
        % function
        
        if model.hrv.include
            [convHRV, ons_secs.hr, verbose] = tapas_physio_create_hrv_regressors(...
                ons_secs, sqpar, model.hrv, verbose);
            
            [model.R, model.R_column_names] = append_regressors(model.R, model.R_column_names, ...
                convHRV, 'HR * CRF');
        end
        
        
        %% 4.4. Create a respiratory volume/time regressor using the respiratory response
        % function
        
        if model.rvt.include
            [convRVT, ons_secs.rvt, verbose] = tapas_physio_create_rvt_regressors(...
                ons_secs, sqpar, model.rvt, verbose);
            
            [model.R, model.R_column_names] = append_regressors(model.R, model.R_column_names, ...
                convRVT, 'RVT * RRF');
        end
        
    end % hasPhyslogFiles
    
    
    %% 4.5. Extract anatomical defined (ROI) principal component regressors
    
    if model.noise_rois.include
        [noise_rois_R, model.noise_rois, verbose] = tapas_physio_create_noise_rois_regressors(...
            model.noise_rois, verbose);
        
        [model.R, model.R_column_names] = append_regressors(model.R, model.R_column_names, ...
            noise_rois_R, 'Noise ROIs');
    end
    
    
    %% 4.6. Load other (physiological) confound regressors
    
    if model.other.include && ~isempty(model.other.input_multiple_regressors)
        [other_R, verbose] = tapas_physio_load_other_multiple_regressors(...
            model.other.input_multiple_regressors, verbose);
        
        [model.R, model.R_column_names] = append_regressors(model.R, model.R_column_names, ...
            other_R, 'Other');
    end
    
    
    %% 4.7. Load and manipulate movement parameters as confound regressors
    
    if model.movement.include && ~isempty(model.movement.file_realignment_parameters)
        [movement_R, model.movement, verbose] = ...
            tapas_physio_create_movement_regressors(model.movement, verbose);
        
        [model.R, model.R_column_names] = append_regressors(model.R, model.R_column_names, ...
            movement_R(:, 1:model.movement.order), 'Movement');
        [model.R, model.R_column_names] = append_regressors(model.R, model.R_column_names, ...
            movement_R(:, model.movement.order+1:end), 'Motion outliers');
    end
    
    
    %% 4.8. Orthogonalisation of regressors ensures numerical stability for
    % otherwise correlated cardiac regressors
    
    [model.R, verbose] = tapas_physio_orthogonalise_physiological_regressors(...
        model.R, model.R_column_names, model.orthogonalise, ...
        verbose);
    
    
    %% 4.9   Save Multiple Regressors file for SPM
    
    physio.save_dir     = save_dir;
    physio.log_files    = log_files;
    physio.preproc      = preproc;
    physio.scan_timing  = scan_timing;
    physio.model        = model;
    physio.verbose      = verbose;
    physio.ons_secs     = ons_secs;
    
    
    % determine file names for output, append slice index, if multiple slices
    % chosen
    if nOnsetSlices > 1
        % save final physio-structure in .mat-file
        [fp, fn, ext] = fileparts(model.output_physio);
        file_output_physio = ...
            fullfile(fp, [fn, sprintf('_slice%03d', onset_slice), ext]);
        
        [fp, fn, ext] = fileparts(model.output_multiple_regressors);
        file_output_multiple_regressors = ...
            fullfile(fp, [fn, sprintf('_slice%03d', onset_slice), ext]);
    else
        file_output_physio = model.output_physio;
        file_output_multiple_regressors = model.output_multiple_regressors;
    end
    
    if ~isempty(model.output_physio)
        save(file_output_physio, 'physio');
    end
    
    
    if isempty(model.R)
        disp(['No model estimated. Only saving read-in log-files data into physio ' ...
            'mat-output-file instead: Check variable physio.ons_secs']);
    else
        [fpfx, fn, fsfx] = fileparts(file_output_multiple_regressors);
        R = model.R;
        names = model.R_column_names;
        
        switch fsfx
            case '.mat'
                save(file_output_multiple_regressors, 'R', 'names');  % SPM understands `names`
            otherwise
                save(file_output_multiple_regressors, 'R', '-ascii', '-double', '-tabs');
        end
    end
    
end % onset_slices



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 5. Save output figures to files
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[physio.verbose] = tapas_physio_print_figs_to_file(physio.verbose);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Helper functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [R, names] = append_regressors(R, names, regressors, name)

R = [R, regressors];
names = [names, repmat({name}, 1, size(regressors, 2))];

end