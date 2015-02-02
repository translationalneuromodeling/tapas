function [physio_out, R, ons_secs] = tapas_physio_main_create_regressors(varargin)
% Main Toolbox Function for preprocessing & modelling from physio-structure
%
% [physio_out, R, ons_secs] = tapas_physio_main_create_regressors(physio)
%
%   OR
%
% [physio_out, R, ons_secs] = tapas_physio_main_create_regressors(...
%    log_files, sqpar, model, thresh, verbose, save_dir);
%
% NOTE: All inputs in physio-structure have to be specified previous to
%       running this function.
%
% IN
%   physio      physio-structure, See also tapas_physio_new
%               OR
%               sub-structures of physio-structure, i.e.
%               log_files, sqpar, thresh, model, verbose, save_dir
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
% $Id: tapas_physio_main_create_regressors.m 670 2015-02-01 19:21:55Z kasperla $
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
    physio.save_dir = varargin{6};
    physio.log_files = varargin{1};
    physio.thresh  = varargin{3};
    physio.sqpar   = varargin{2};
    physio.model   = varargin{4};
    physio.verbose = varargin{5};
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
thresh      = physio.thresh;
sqpar       = physio.sqpar;
model       = physio.model;
verbose     = physio.verbose;


hasPhaseLogfile = strcmpi(log_files.vendor, 'CustomPhase');

if ~hasPhaseLogfile
    
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% 1. Read in vendor-specific physiological log-files
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    [ons_secs.c, ons_secs.r, ons_secs.t, ons_secs.cpulse, ons_secs.acq_codes, ...
        verbose] = tapas_physio_read_physlogfiles(...
        log_files, thresh.cardiac.modality, verbose);
    
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
    
    if verbose.level >= 2
        verbose.fig_handles(end+1) = tapas_physio_plot_raw_physdata(ons_secs);
    end
    
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% 2. Create scan timing nominally or from logfile
    % (Philips: via gradient time-course; Siemens (NEW): from tics)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    useNominal = isempty(thresh.scan_timing) || ...
        strcmpi(thresh.scan_timing.method, 'nominal');
    if useNominal
        [VOLLOCS, LOCS] = ...
            tapas_physio_create_nominal_scan_timing(ons_secs.t, sqpar, ...
            log_files.align_scan);
    else
        switch thresh.scan_timing.method
            case {'gradient', 'gradient_log'}
                [VOLLOCS, LOCS, verbose] = ...
                    tapas_physio_create_scan_timing_from_gradients_philips( ...
                    log_files, thresh.scan_timing, sqpar, verbose);
            case {'gradient_auto', 'gradient_log_auto'}
                [VOLLOCS, LOCS, verbose] = ...
                    tapas_physio_create_scan_timing_from_gradients_auto_philips( ...
                    log_files, thresh.scan_timing, sqpar, verbose);
            case 'scan_timing_log'
                [VOLLOCS, LOCS, verbose] = ...
                    tapas_physio_create_scan_timing_from_tics_siemens( ...
                    ons_secs.t, log_files, verbose);
        end
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
        % thresh.cardiac.modality = 'OXY'; % 'ECG' or 'OXY' (for pulse oximetry)
        %% initial pulse select via load from logfile or autocorrelation with 1
        %% cardiac pulse
        switch thresh.cardiac.initial_cpulse_select.method
            case {'load_from_logfile', ''}
                % do nothing
            otherwise
                % run one of the various cardiac pulse detection algorithms
                minCardiacCycleSamples = floor((1/(90/60)/dt));
                [ons_secs.cpulse, verbose] = tapas_physio_get_cardiac_pulses(ons_secs.t, ons_secs.c, ...
                    thresh.cardiac.initial_cpulse_select, thresh.cardiac.modality, minCardiacCycleSamples, verbose);
        end
        
        
        %% post-hoc: hand pick additional cardiac pulses or load from previous
        %% time
        switch thresh.cardiac.posthoc_cpulse_select.method
            case {'manual'}
                % additional manual fill-in of more missed pulses
                [ons_secs, outliersHigh, outliersLow] = ...
                    tapas_physio_correct_cardiac_pulses_manually(ons_secs, ...
                    thresh.cardiac.posthoc_cpulse_select);
            case {'load'}
                hasPosthocLogFile = exist(thresh.cardiac.posthoc_cpulse_select.file, 'file') || ...
                    exist([thresh.cardiac.posthoc_cpulse_select.file '.mat'], 'file');
                
                if hasPosthocLogFile % load or set selection to manual, if no file exists
                    osload = load(thresh.cardiac.posthoc_cpulse_select.file, 'ons_secs');
                    ons_secs = osload.ons_secs;
                else
                    [ons_secs, outliersHigh, outliersLow] = ...
                        tapas_physio_correct_cardiac_pulses_manually(ons_secs,...
                        thresh.cardiac.posthoc_cpulse_select);
                end
            case {'off', ''}
        end
        
    end
    
    
    [ons_secs, sqpar] = tapas_physio_crop_scanphysevents_to_acq_window(ons_secs, sqpar);
    
    if hasRespData
        % filter respiratory signal
        ons_secs.fr = tapas_physio_filter_respiratory(ons_secs.r, ...
            dt, doNormalize);
    end
    
    if verbose.level >= 2
        verbose.fig_handles(end+1) = ...
            tapas_physio_plot_cropped_phys_to_acqwindow(ons_secs, sqpar);
    end
    
    if verbose.level >= 1
        verbose.fig_handles(end+1) = ...
            tapas_physio_plot_raw_physdata_diagnostics(ons_secs.cpulse, ...
            ons_secs.r, thresh.cardiac.posthoc_cpulse_select, verbose.level, ...
            ons_secs.t, ons_secs.c);
    else % without figure creation    else
        tapas_physio_plot_raw_physdata_diagnostics(ons_secs.cpulse, ...
            ons_secs.r, thresh.cardiac.posthoc_cpulse_select, 0);
    end
    
    
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
%% 4. Create RETROICOR/response function regressors for GLM
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if any(strfind(upper(model.type),'RETROICOR'))
    [cardiac_sess, respire_sess, mult_sess, ons_secs, verbose] = ...
        tapas_physio_create_retroicor_regressors(ons_secs, sqpar, ...
        model.order, verbose);
else
    cardiac_sess = [];
    respire_sess = [];
    mult_sess = [];
end


% Create a heart-rate variability regressor using the cardiac response
% function

if any(strfind(upper(model.type),'HRV'))
    [convHRV, ons_secs.hr, verbose] = tapas_physio_create_hrv_regressor(...
        ons_secs, sqpar, verbose);
else
    convHRV = [];
end


% Create a respiratory volume/time regressor using the respiratory response
% function

if any(strfind(upper(model.type),'RVT'))
    [convRVT, ons_secs.rvt, verbose] = tapas_physio_create_rvt_regressor(...
        ons_secs, sqpar, verbose);
else
    convRVT = [];
end


% 4.1 Load other confound regressors, e.g. realigment parameters

if isfield(model, 'input_other_multiple_regressors') && ~isempty(model.input_other_multiple_regressors)
    input_R = tapas_physio_load_other_multiple_regressors(model.input_other_multiple_regressors);
else
    input_R = [];
end

input_R = [input_R, convHRV, convRVT];


% 4.2   Orthogonalisation of regressors ensures numerical stability for
%       otherwise correlated cardiac regressors

[R, verbose] = tapas_physio_orthogonalise_physiological_regressors(cardiac_sess, respire_sess, ...
    mult_sess, input_R, model.order.orthogonalise, verbose);


% 4.3   Save Multiple Regressors file for SPM

model.R = R;

physio_out.save_dir     = save_dir;
physio_out.log_files    = log_files;
physio_out.thresh       = thresh;
physio_out.sqpar        = sqpar;
physio_out.model        = model;
physio_out.verbose      = verbose;
physio_out.ons_secs     = ons_secs;


switch lower(model.type)
    case 'none'
        disp('No model estimated. Saving read log-files data into output-file instead: Check variable physio.ons_secs');
        if ~isempty(model.output_multiple_regressors)
            [fpfx, fn] = fileparts(model.output_multiple_regressors);
            save(fullfile(fpfx, [fn '.mat']), 'physio_out');
        end
        
    otherwise
        [fpfx, fn, fsfx] = fileparts(model.output_multiple_regressors);
        
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

if isfield(verbose, 'fig_output_file') && ~isempty(verbose.fig_output_file)
    tapas_physio_print_figs_to_file(verbose);
end
