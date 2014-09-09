function [physio_out, R, ons_secs] = tapas_physio_main_create_regressors(log_files, ...
    thresh, sqpar, model, verbose)
% RETROICOR - regressor creation based on Glover, G. MRM 44, 2000
%
% USAGE
% [physio_out, R, ons_secs] = tapas_physio_main_create_regressors(physio)
%
%   OR
%
% [physio_out, R, ons_secs] = tapas_physio_main_create_regressors(log_files, ...
%    thresh, sqpar, model, verbose)
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
% $Id: tapas_physio_main_create_regressors.m 235 2013-08-19 16:28:07Z kasperla $
%


%% 0. set default parameters
if ~nargin
    error('Please specify a PhysIO-object as input to this function. See tapas_physio_new');
end


if nargin == 1 % assuming sole PhysIO-object as input
    physio      = log_files; % first argument of function
    log_files = physio.log_files;
    thresh  = physio.thresh;
    sqpar   = physio.sqpar;
    model   = physio.model;
    verbose = physio.verbose;
end
    


%% 1. Read in vendor-specific physiological log-files
[ons_secs.c, ons_secs.r, ons_secs.t, ons_secs.cpulse] = ...
    tapas_physio_read_physlogfiles(log_files, thresh.cardiac.modality);

if verbose.level >= 1
    verbose.fig_handles(end+1) = tapas_physio_plot_raw_physdata(ons_secs);
end


%% 2. Create scan timing nominally or from gradient time-course
% the latter is only necessary, if no patch is used and therefore no scan event
% triggers are written into the last column of the scanphyslog-file
if isempty(thresh.scan_timing)
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


%% initial pulse select via load from logfile or autocorrelation with 1
%% cardiac pulse
switch thresh.cardiac.initial_cpulse_select.method
    case {'manual', 'load'}
    [ons_secs.cpulse, verbose] = tapas_physio_get_cardiac_pulses(ons_secs.t, ons_secs.c, ...
        thresh.cardiac.initial_cpulse_select, thresh.cardiac.modality, verbose); 
    case {'load_from_logfile', ''}
end

%% post-hoc: hand pick additional cardiac pulses or load from previous
%% time
switch thresh.cardiac.posthoc_cpulse_select.method
    case {'manual'}
    % additional manual fill-in of more missed pulses
    [ons_secs, outliersHigh, outliersLow] = tapas_physio_correct_cardiac_pulses_manually(ons_secs,thresh.cardiac.posthoc_cpulse_select);
    case {'load'}
        osload = load(thresh.cardiac.posthoc_cpulse_select.file, 'ons_secs');
        ons_secs = osload.ons_secs;
    case {'off', ''}
end



[ons_secs, sqpar] = tapas_physio_crop_scanphysevents_to_acq_window(ons_secs, sqpar);
if verbose.level >= 1
    verbose.fig_handles(end+1) = ...
        tapas_physio_plot_cropped_phys_to_acqwindow(ons_secs, sqpar);
end

if verbose.level >= 2
    verbose.fig_handles(end+1) = ...
        tapas_physio_plot_raw_physdata_diagnostics(ons_secs.cpulse, ons_secs.r, thresh.cardiac.posthoc_cpulse_select);
end

%% 4. Create RETROICOR regressors for SPM
switch upper(model.type)
    case 'RETROICOR'
        [cardiac_sess, respire_sess, mult_sess, verbose, ...
            c_sample_phase, r_sample_phase] = ...
            tapas_physio_create_retroicor_regressors(ons_secs, sqpar, thresh, ...
            model.order, verbose);
            ons_secs.c_sample_phase = c_sample_phase;
            ons_secs.r_sample_phase = r_sample_phase;
    otherwise
        error('Please valid specify model.type');
end

% 4.1.  Load other confound regressors, e.g. realigment parameters
if isfield(model, 'input_other_multiple_regressors') && ~isempty(model.input_other_multiple_regressors)
    input_R = tapas_physio_load_other_multiple_regressors(model.input_other_multiple_regressors);
else
    input_R = [];
end


% 4.2   Orthogonalisation of regressors ensures numerical stability for 
%       otherwise correlated cardiac regressors
[R, verbose] = tapas_physio_orthogonalise_physiological_regressors(cardiac_sess, respire_sess, ...
    mult_sess, input_R, model.order.orthogonalise, verbose);

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

physio_out.log_files    = log_files;
physio_out.thresh       = thresh;
physio_out.sqpar        = sqpar;
physio_out.model        = model;
physio_out.verbose      = verbose;
physio_out.ons_secs     = ons_secs;

