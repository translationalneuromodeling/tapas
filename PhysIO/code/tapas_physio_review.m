function verbose = tapas_physio_review(physio, newVerboseLevel)
% Reviews performance of PhysIO (recreating output plots and text)
% after running tapas_physio_main_create_regressors
%
%   verbose = tapas_physio_review(physio, newVerboseLevel)
%
% NOTE: Change physio.verbose.level before running this function to get
%       additonal output plots not seen during executing of the main-function
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
%   verbose.fig_handles     [nFigures,1] vector of figure handles created
%
% EXAMPLE
%   tapas_physio_review
%
%   See also
%
% Author: Lars Kasper
% Created: 2016-10-27
% Copyright (C) 2016 TNU, Institute for Biomedical Engineering,
%                    University of Zurich and ETH Zurich.
%
% This file is part of the TAPAS PhysIO Toolbox, which is released under the terms of the GNU General Public
% License (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.
%
% $Id: teditRETRO.m 775 2015-07-17 10:52:58Z kasperla $

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

verbose = tapas_physio_plot_raw_physdata(ons_secs.raw, verbose);

% tapas_physio_get_onsets_from_locs -> create plot function out of
% sub-function
% tapas_physio_get_cardiac_pulses_auto_matched -> subfunction for plot, only called
% if in this sub-branch

if verbose.level >= 2
    verbose.fig_handles(end+1) = ...
        tapas_physio_plot_cropped_phys_to_acqwindow(ons_secs, sqpar);
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


%% Create mock SPM to determine columns with get_regressor-function

SPM.Sess = 1;
nRegressors = size(model.R,2);
s = 1;

SPM.xX.name = cellfun(@(iCard) ['Sn(' int2str(s) ') R' int2str(iCard)], ...
    num2cell(1:nRegressors), 'UniformOutput', false);

[colPhys, colCard, colResp, colMult, colHRV, colRVT, colRois, colMove, colAll] = ...
    tapas_physio_check_get_regressor_columns(SPM, model);

if model.retroicor.include
    R = model.R(:,[colCard,colResp,colMult]);
    hasCardiacData = ~isempty(ons_secs.c);
    hasRespData = ~isempty(ons_secs.r);
    verbose.fig_handles(end+1) = ...
        tapas_physio_plot_retroicor_regressors(R, model.retroicor.order, hasCardiacData, ...
        hasRespData);
end


if model.movement.include
    censoring = model.movement.censoring;
    quality_measures = model.movement.quality_measures;
    switch lower(model.movement.censoring_method)
        case 'fd'
            verbose.fig_handles(end+1) = tapas_physio_plot_movement_outliers_fd(rp, ...
                quality_measures, censoring, movement.censoring_threshold);
        case 'maxval'
            verbose.fig_handles(end+1) = tapas_physio_plot_movement_outliers_maxval(rp, ...
                quality_measures, censoring, movement.censoring_threshold);
    end
end

%% tapas_physio_create_hrv_regressors, tapas_physio_create_rvt_regressors
% tapas_physio_create_noise_rois_regressors
% => create functions out of inline-plotting

%% TODO: replace this call by just using the plot-subpart of the
% orthogonalization
cardiac_sess = model.R(:,colCard);
respire_sess = model.R(:,colResp);
mult_sess = model.R(:,colMult);

[R, verbose] = tapas_physio_orthogonalise_physiological_regressors(...
    cardiac_sess, respire_sess, ...
    mult_sess, model.R, model.orthogonalise, verbose);
