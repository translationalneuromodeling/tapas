function [physio_out, R, ons_secs] = tapas_physio_main_create_regressors(...
    log_files, sqpar, model, thresh, verbose, save_dir)
% RETROICOR - regressor creation based on Glover, G. MRM 44, 2000
%
% USAGE
% [physio_out, R, ons_secs] = tapas_physio_main_create_regressors(physio)
%
%   OR
%
% [physio_out, R, ons_secs] = tapas_physio_main_create_regressors(...
%    log_files, sqpar, model, thresh, verbose, save_dir);
%
%------------------------------------------------------------------------
% IN
%   physio
%
% OUT
%   physio_out
%   R
%   ons_secs
%
% See also tapas_physio_new
%
% -------------------------------------------------------------------------
% Lars Kasper, August 2011
%
% Copyright (C) 2013, Institute for Biomedical Engineering, ETH/Uni Zurich.
%
% This file is part of the PhysIO toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.
%
% $Id: tapas_physio_main_create_regressors.m 524 2014-08-13 16:21:56Z kasperla $
%


%% 0. set default parameters
if ~nargin
    error('Please specify a PhysIO-object as input to this function. See tapas_physio_new');
end


if nargin == 1 % assuming sole PhysIO-object as input
    physio      = log_files; % first argument of function
else % assemble physio-structure
    physio = tapas_physio_new();
    physio.save_dir = save_dir;
    physio.log_files = log_files;
    physio.thresh  = thresh;
    physio.sqpar   = sqpar;
    physio.model   = model;
    physio.verbose = verbose;
end

% fill up empty parameters
physio = tapas_physio_fill_empty_parameters(physio);

% replace cellstrings
physio = tapas_physio_cell2char(physio);

% prepend absolute directories - save_dir
physio = tapas_physio_prepend_absolute_paths(physio);

% set sub-structures for readability; NOTE: copy by value, physio-structure not
% updated!
save_dir = physio.save_dir;
log_files = physio.log_files;
thresh  = physio.thresh;
sqpar   = physio.sqpar;
model   = physio.model;
verbose = physio.verbose;

%% 1. Read in vendor-specific physiological log-files
[ons_secs.c, ons_secs.r, ons_secs.t, ons_secs.cpulse, verbose] = ...
    tapas_physio_read_physlogfiles(log_files, thresh.cardiac.modality, ...
    verbose);

% since resampling might have occured, dt is recalculated
dt = ons_secs.t(2)-ons_secs.t(1); 

hasCardiacData = ~isempty(ons_secs.c);
hasRespData = ~isempty(ons_secs.r);

if verbose.level >= 2
    verbose.fig_handles(end+1) = tapas_physio_plot_raw_physdata(ons_secs);
end


%% 2. Create scan timing nominally or from gradient time-course
% the latter is only necessary, if no patch is used and therefore no scan event
% triggers are written into the last column of the scanphyslog-file
useNominal = isempty(thresh.scan_timing) || ...
    strcmpi(thresh.scan_timing.method, 'nominal');
if useNominal
    [VOLLOCS, LOCS] = tapas_physio_create_nominal_scan_timing(ons_secs.t, sqpar);
else
    [VOLLOCS, LOCS, verbose] = tapas_physio_create_scan_timing_from_gradients_philips( ...
        log_files, thresh.scan_timing, sqpar, verbose);
end

[ons_secs.svolpulse, ons_secs.spulse, ons_secs.spulse_per_vol, verbose] = ...
    tapas_physio_get_onsets_from_locs(ons_secs.t, VOLLOCS, LOCS, sqpar, verbose);


%% 3. Extract and check physiological parameters (onsets)
% plot whether physdata is alright or events are missing (too low/high
% heart rate? breathing amplitude overshoot?)
if hasCardiacData
    % thresh.cardiac.modality = 'OXY'; % 'ECG' or 'OXY' (for pulse oximetry)
    
    %% initial pulse select via load from logfile or autocorrelation with 1
    %% cardiac pulse
    switch thresh.cardiac.initial_cpulse_select.method
        case {'load_from_logfile', ''}
            % do nothing
        otherwise
            % run one of the various cardiac pulse detection algorithms
            [ons_secs.cpulse, verbose] = tapas_physio_get_cardiac_pulses(ons_secs.t, ons_secs.c, ...
                thresh.cardiac.initial_cpulse_select, thresh.cardiac.modality, [], verbose);
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
            hasPostocLogFile = exist(thresh.cardiac.posthoc_cpulse_select.file, 'file') || ...
                exist([thresh.cardiac.posthoc_cpulse_select.file '.mat'], 'file');
            
            if hasPostocLogFile % load or set selection to manual, if no file exists
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
if verbose.level >= 2
    verbose.fig_handles(end+1) = ...
        tapas_physio_plot_cropped_phys_to_acqwindow(ons_secs, sqpar);
end

if verbose.level >= 1
    verbose.fig_handles(end+1) = ...
        tapas_physio_plot_raw_physdata_diagnostics(ons_secs.cpulse, ...
        ons_secs.r, thresh.cardiac.posthoc_cpulse_select, verbose.level, ...
    ons_secs.t, ons_secs.c);
else % without figure creation
     tapas_physio_plot_raw_physdata_diagnostics(ons_secs.cpulse, ...
        ons_secs.r, thresh.cardiac.posthoc_cpulse_select, 0);
end

if hasRespData
    % filter respiratory signal
    ons_secs.fr = tapas_physio_filter_respiratory(ons_secs.r, ...
        dt);
end

%% 4. Create RETROICOR/response function regressors for SPM
if any(strfind(upper(model.type),'RETROICOR'))
    [cardiac_sess, respire_sess, mult_sess, ons_secs, verbose] = ...
        tapas_physio_create_retroicor_regressors(ons_secs, sqpar, ...
        model.order, verbose);
else
    cardiac_sess = [];
    respire_sess = [];
    mult_sess = [];
end

% create a heart-rate variability regressor using the cardiac response
% function
if any(strfind(upper(model.type),'HRV'))
    [convHRV, ons_secs.hr, verbose] = tapas_physio_create_hrv_regressor(...
        ons_secs, sqpar, verbose);
else
    convHRV = [];
end

% create a respiratory volume/time regressor using the cardiac response
% function
if any(strfind(upper(model.type),'RVT'))
    [convRVT, ons_secs.rvt, verbose] = tapas_physio_create_rvt_regressor(...
        ons_secs, sqpar, verbose);
else
    convRVT = [];
end

% 4.1.  Load other confound regressors, e.g. realigment parameters
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

if isempty(R)
    error('Please specify valid model.type');
end


% 4.3   Save Multiple Regressors file for SPM
[fpfx, fn, fsfx] = fileparts(model.output_multiple_regressors);

switch fsfx
    case '.mat'
        save(model.output_multiple_regressors, 'R');
    otherwise
        save(model.output_multiple_regressors, 'R', '-ascii', '-double', '-tabs');
end
model.R = R;

if isfield(verbose, 'fig_output_file') && ~isempty(verbose.fig_output_file)
    tapas_physio_print_figs_to_file(verbose);
end

physio_out.save_dir     = save_dir;
physio_out.log_files    = log_files;
physio_out.thresh       = thresh;
physio_out.sqpar        = sqpar;
physio_out.model        = model;
physio_out.verbose      = verbose;
physio_out.ons_secs     = ons_secs;

