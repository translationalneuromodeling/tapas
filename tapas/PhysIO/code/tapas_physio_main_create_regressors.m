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
%
% Author: Lars Kasper
% Created: 2011-08-01
% Copyright (C) 2013 TNU, Institute for Biomedical Engineering, University of Zurich and ETH Zurich.
%
% This file is part of the PhysIO toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.
%
% $Id: tapas_physio_main_create_regressors.m 815 2015-08-18 20:52:47Z kasperla $
%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 0. Set Default parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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
sqpar       = scan_timing.sqpar;

hasPhaseLogfile = strcmpi(log_files.vendor, 'CustomPhase');

if ~hasPhaseLogfile
    
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% 1. Read in vendor-specific physiological log-files
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    [ons_secs.c, ons_secs.r, ons_secs.t, ons_secs.cpulse, ons_secs.acq_codes, ...
        verbose] = tapas_physio_read_physlogfiles(...
        log_files, preproc.cardiac.modality, verbose);
    
    % also: normalize cardiac/respiratory data, if wanted
    doNormalize = true;
    if doNormalize
        maxAbsC = max(abs(ons_secs.c));
        if ~isempty(maxAbsC)
            ons_secs.c_scaling = maxAbsC ;
            ons_secs.c = ons_secs.c/maxAbsC;
        end
        
        maxAbsR = max(abs(ons_secs.r));
        if ~isempty(maxAbsR)
            ons_secs.r_scaling = maxAbsR ;
            ons_secs.r = ons_secs.r/maxAbsR;
        end
        
    end
    
    % since resampling might have occured, dt is recalculated
    dt = ons_secs.t(2) - ons_secs.t(1);
    
    hasCardiacData = ~isempty(ons_secs.c);
    hasRespData = ~isempty(ons_secs.r);
    
   
    verbose = tapas_physio_plot_raw_physdata(ons_secs, verbose);
    
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% 2. Create scan timing nominally or from logfile
    % (Philips: via gradient time-course; Siemens (NEW): from tics)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
    switch lower(scan_timing.sync.method)
        case 'nominal'
            [VOLLOCS, LOCS] = ...
                tapas_physio_create_nominal_scan_timing(ons_secs.t, ...
                sqpar, log_files.align_scan);
        case {'gradient', 'gradient_log'}
            [VOLLOCS, LOCS, verbose] = ...
                tapas_physio_create_scan_timing_from_gradients_philips( ...
                log_files, scan_timing, verbose);
        case {'gradient_auto', 'gradient_log_auto'}
            [VOLLOCS, LOCS, verbose] = ...
                tapas_physio_create_scan_timing_from_gradients_auto_philips( ...
                log_files, scan_timing, verbose);
        case 'scan_timing_log'
            [VOLLOCS, LOCS, verbose] = ...
                tapas_physio_create_scan_timing_from_tics_siemens( ...
                ons_secs.t, log_files, verbose);
    end
    
    
    % remove arbitrary offset in time vector now, since all timings have now
    % been aligned to ons_secs.t
    ons_secs.t = ons_secs.t - ons_secs.t(1);
    
    [ons_secs.svolpulse, ons_secs.spulse, ons_secs.spulse_per_vol, verbose] = ...
        tapas_physio_get_onsets_from_locs(...
        ons_secs.t, VOLLOCS, LOCS, sqpar, verbose);
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% 3. Extract and preprocess physiological data, crop to scan aquisition
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    if hasCardiacData
        % preproc.cardiac.modality = 'OXY'; % 'ECG' or 'OXY' (for pulse oximetry)
        %% initial pulse select via load from logfile or autocorrelation with 1
        %% cardiac pulse
        switch preproc.cardiac.initial_cpulse_select.method
            case {'load_from_logfile', ''}
                % do nothing
            otherwise
                % run one of the various cardiac pulse detection algorithms
                minCardiacCycleSamples = floor((1/(90/60)/dt));
                [ons_secs.cpulse, verbose] = ...
                    tapas_physio_get_cardiac_pulses(ons_secs.t, ons_secs.c, ...
                    preproc.cardiac.initial_cpulse_select, ...
                    preproc.cardiac.modality, minCardiacCycleSamples, verbose);
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
        
    end
    
    
    [ons_secs, sqpar, verbose] = tapas_physio_crop_scanphysevents_to_acq_window(...
        ons_secs, sqpar, verbose);
    scan_timing.sqpar = sqpar;
    
    if hasRespData
        % filter respiratory signal
        ons_secs.fr = tapas_physio_filter_respiratory(ons_secs.r, ...
            dt, doNormalize);
    end
    
    if verbose.level >= 2
        verbose.fig_handles(end+1) = ...
            tapas_physio_plot_cropped_phys_to_acqwindow(ons_secs, sqpar);
    end
    
    [verbose, ons_secs.c_outliers_low, ons_secs.c_outliers_high, ...
        ons_secs.r_hist] = ...
        tapas_physio_plot_raw_physdata_diagnostics(ons_secs.cpulse, ...
        ons_secs.r, preproc.cardiac.posthoc_cpulse_select, verbose, ...
        ons_secs.t, ons_secs.c);
    
else
    % Phase data saved in log-file already
    % Read logged phases into object directly
    load(log_files.cardiac)
    
    ons_secs.c_sample_phase = c_phase_probe_regressors(...
        (140+sqpar.onset_slice):(sqpar.Nslices):end);
    
    load(log_files.respiratory);
    ons_secs.r_sample_phase = r_phase_probe_regressors(...
        (sqpar.onset_slice):(sqpar.Nslices):end);
    
end




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 4. Create physiological noise model regressors for GLM
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if model.retroicor.include
    [cardiac_sess, respire_sess, mult_sess, ons_secs, ...
        model.retroicor.order, verbose] = ...
        tapas_physio_create_retroicor_regressors(ons_secs, sqpar, ...
        model.retroicor.order, verbose);
else
    cardiac_sess = [];
    respire_sess = [];
    mult_sess = [];
end


% Create a heart-rate variability regressor using the cardiac response
% function

if model.hrv.include % TODO: include delays!
    [convHRV, ons_secs.hr, verbose] = tapas_physio_create_hrv_regressors(...
        ons_secs, sqpar, model.hrv, verbose);
else
    convHRV = [];
end


% Create a respiratory volume/time regressor using the respiratory response
% function

if model.rvt.include
    [convRVT, ons_secs.rvt, verbose] = tapas_physio_create_rvt_regressors(...
        ons_secs, sqpar, model.rvt, verbose);
else
    convRVT = [];
end

% Extract anatomical defined (ROI) principal component regressors

if model.noise_rois.include
    [noise_rois_R, model.noise_rois, verbose] = tapas_physio_create_noise_rois_regressors(...
        model.noise_rois, verbose);
else
    noise_rois_R = [];
end


% Load other (physiological) confound regressors

if model.other.include && ~isempty(model.other.input_multiple_regressors)
    [other_R, verbose] = tapas_physio_load_other_multiple_regressors(...
        model.other.input_multiple_regressors, verbose);
else
    other_R = [];
end

% load and manipulate movement parameters as confound regressors
if model.movement.include && ~isempty(model.movement.file_realignment_parameters)
     [movement_R, verbose] = tapas_physio_create_movement_regressors(...
         model.movement, verbose);
else
    movement_R = [];
end




R = [convHRV, convRVT, noise_rois_R, movement_R, other_R ];


% Orthogonalisation of regressors ensures numerical stability for
% otherwise correlated cardiac regressors

[R, verbose] = tapas_physio_orthogonalise_physiological_regressors(...
    cardiac_sess, respire_sess, ...
    mult_sess, R, model.orthogonalise, verbose);


% 4.3   Save Multiple Regressors file for SPM

model.R = R;

physio.save_dir     = save_dir;
physio.log_files    = log_files;
physio.preproc      = preproc;
physio.scan_timing  = scan_timing;
physio.model        = model;
physio.verbose      = verbose;
physio.ons_secs     = ons_secs;

% save final physio-structure in .mat-file
if ~isempty(model.output_physio)
    save(model.output_physio, 'physio');
end


if isempty(R)
    disp(['No model estimated. Saving read log-files data into physio ' ...
        'output-file instead: Check variable physio.ons_secs']);
else
    [fpfx, fn, fsfx] = fileparts(model.output_multiple_regressors);
    
    % TODO: slice-wise saving here...
    % indSlice = physio.scan_timing.sqpar.onset_slice
    
    switch fsfx
        case '.mat'
            save(model.output_multiple_regressors, 'R');
        otherwise
            save(model.output_multiple_regressors, 'R', '-ascii', '-double', '-tabs');
    end
end




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 5. Save output figures to files
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[physio.verbose] = tapas_physio_print_figs_to_file(physio.verbose);

