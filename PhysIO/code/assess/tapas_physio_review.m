function verbose = tapas_physio_review(physio, newVerboseLevel)
% Reviews performance of PhysIO (recreating output plots and text)
% after running tapas_physio_main_create_regressors
%
%   verbose = tapas_physio_review(physio, newVerboseLevel)
%
% NOTE: Change physio.verbose.level before running this function to get
%       additonal output plots not seen during executing of the main-function
%
% NOTE1: Change the following flags prior to running this function in order
% to control the output figure behavior.
%   physio.verbose.show_figs = false; (default true)
%   physio.verbose.save_figs = true; (default false)
%   physio.verbose.close_figs = true; (default false)
% An example use case is to disable figure outputs when running
% tapas_physio_main_create_regressors (by setting physio.verbose.level = 0)
% and then running tapas_physio_review in order to generate the figures.
% Figures can, for example, be generated and saved in the background
% (show_figs = false, save_figs = true) or displayed and not saved
% (show_figs = true, save_figs = false).
%
% NOTE2: This is not an exact copy of the plotting behavior within
% tapas_physio_main_create_regressors (yet). The most important plots for
% diagnostics should show up, though.
%
% IN
%   physio      physio-structure (or mat-file) saved after successful run
%               of tapas_physio_main_create_regressors
%               See also tapas_physio_new
%   newVerboseLevel
%               visual verbosity level to (re-) create report plots
%               default: 2
% OUT
%   several output plots and command line information on the toolbox
%   performance
%
%   verbose.fig_handles     [1, nFigures] vector of figure handles created
%
% EXAMPLE
%   tapas_physio_review
%
%   See also

% Author: Lars Kasper
% Created: 2016-10-27
% Copyright (C) 2016 TNU, Institute for Biomedical Engineering,
%                    University of Zurich and ETH Zurich.
%
% This file is part of the TAPAS PhysIO Toolbox, which is released under the terms of the GNU General Public
% License (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.



if ischar(physio) % file name
    load(physio);
elseif iscell(physio) % cell string
    load(physio{1});
end

if nargin < 2
    newVerboseLevel = 2;
end

ons_secs    = physio.ons_secs;
save_dir    = physio.save_dir;
log_files   = physio.log_files;
preproc     = physio.preproc;
scan_timing = physio.scan_timing;
sqpar       = scan_timing.sqpar;
sync        = scan_timing.sync;
model       = physio.model;
verbose     = physio.verbose;

% Compatibility with old versions
if ~isfield(model, 'R_column_names')
    disp('Reconstructing regressor names...')
    model.R_column_names = tapas_physio_guess_regressor_names(model, model.R);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 1. Write out all information from process log
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fid = 1;
nLogs = numel(verbose.process_log);

for iLog = 1:nLogs
    fprintf(fid, verbose.process_log{iLog});
    fprintf(fid,'\n\n');
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 2. now re-call all output plot functions in tapas_physio_main_create_regressors in
%     same order
%     TODO: how to call within-function plots?
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% overwrite old verbose
verbose.level       = newVerboseLevel;
verbose.fig_handles = [];
verbose.process_log = {};

% Set default for verbose.show_figs, verbose.save_figs and verbose.close_figs
% if they are empty or if the fields do not exist
% show_figs default = true (i.e. do show)
if ~isfield(verbose, 'show_figs') || isempty(verbose.show_figs)
    verbose.show_figs = true;
end
% save_figs default = false (i.e. do not save)
if ~isfield(verbose, 'save_figs') || isempty(verbose.save_figs)
    verbose.save_figs = false;
end
% close_figs default = false (i.e. do not close)
if ~isfield(verbose, 'close_figs') || isempty(verbose.close_figs)
    verbose.close_figs = false;
end

verbose = tapas_physio_plot_raw_physdata(ons_secs.raw, verbose);

% tapas_physio_get_onsets_from_locs -> create plot function out of
% sub-function
% tapas_physio_get_cardiac_pulses_auto_matched -> subfunction for plot, only called
% if in this sub-branch

if verbose.level >= 2
    verbose.fig_handles(end+1) = ...
        tapas_physio_plot_cropped_phys_to_acqwindow(ons_secs, sqpar, verbose);
end

[verbose, ons_secs.c_outliers_low, ons_secs.c_outliers_high, ...
    ons_secs.r_hist] = ...
    tapas_physio_plot_raw_physdata_diagnostics(ons_secs.cpulse, ...
    ons_secs.r, preproc.cardiac.posthoc_cpulse_select, verbose, ...
    ons_secs.t, ons_secs.c);

% in tapas_physio_create_retroicor_regressors:
% tapas_physio_get_respiratory_phase
% function fh = plot_traces(pulset, rsampint, rout, resp_max, ...
%   cumsumh, sumh, h, npulse, dpulse, rphase)


%% RETROICOR

if model.retroicor.include
    retroicor = model.R(:, contains(model.R_column_names, 'RETROICOR', 'IgnoreCase', true));
    hasCardiacData = ~isempty(ons_secs.c);
    hasRespData = ~isempty(ons_secs.r);
    verbose.fig_handles(end+1) = ...
        tapas_physio_plot_retroicor_regressors(retroicor, model.retroicor.order, hasCardiacData, ...
        hasRespData, verbose);
end


%% Movement

if model.movement.include
    rp = model.movement.rp;
    quality_measures = model.movement.quality_measures;
    censoring = model.movement.censoring;
    censoring_threshold = model.movement.censoring_threshold;
    switch lower(model.movement.censoring_method)
        case 'fd'
            verbose.fig_handles(end+1) = tapas_physio_plot_movement_outliers_fd( ...
                rp, quality_measures, censoring, censoring_threshold, verbose);
        case 'maxval'
            verbose.fig_handles(end+1) = tapas_physio_plot_movement_outliers_maxval( ...
                rp, quality_measures, censoring, censoring_threshold, verbose);
    end
end

%% tapas_physio_create_hrv_regressors, tapas_physio_create_rvt_regressors
% tapas_physio_create_noise_rois_regressors
% => create functions out of inline-plotting

%% Overall regressors

[R, verbose] = tapas_physio_orthogonalise_physiological_regressors(...
    model.R, model.R_column_names, model.orthogonalise, verbose);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Save output figures to files - if specified
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if verbose.save_figs
    [verbose] = tapas_physio_print_figs_to_file(verbose, save_dir);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Close figures - if specified
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This logic assumes that only saved figures will have to be closed (if
% specified), since displaying and again closing unsaved figures will not
% serve any purpose.
if verbose.save_figs && verbose.close_figs
    [verbose] = tapas_physio_close_figs(verbose);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end
