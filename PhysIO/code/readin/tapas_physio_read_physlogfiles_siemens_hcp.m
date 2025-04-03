function [c, r, t, cpulse, acq_codes, verbose] = tapas_physio_read_physlogfiles_siemens_hcp(...
    log_files, cardiac_modality, verbose, varargin)
% Reads in preprocessed Human Connectome Project (HCP) Physiology Data
% NOTE: The physiological log-files downloaded as part of the HCP with the
% name format  *_Physio_log.txt are already preprocessed into a simple 3-column
% text format. This format can be read with this function.
% Importantly, these logfiles do NOT match the HCP's own specification on their
% website (https://wiki.humanconnectome.org/display/PublicData/Understanding+Timing+Information+in+HCP+Physiological+Monitoring+Files)
% - which would be a classical Siemens (VB) logfile format (*.ecg,*.resp,*.puls)
% If you happen to have these raw logfiles from the HCP, go with the
% classical "Siemens" format selection of PhysIO.
%
% [c, r, t, cpulse, acq_codes, verbose] = tapas_physio_read_physlogfiles_siemens_hcp(...
%    log_files, cardiac_modality, verbose, varargin)
%
% IN    log_files
%       .log_cardiac        contains ECG or pulse oximeter time course
%                           for Siemens_HCP: *Physio_log.txt
%       .log_respiration    contains breathing belt amplitude time course
%                           for Siemens_HCP: *Physio_log.txt (same as
%                           cardiac log file!)
%       .sampling_interval  1 entry: sampling interval (seconds)
%                           for both log files
%                           2 entries: 1st entry sampling interval (seconds)
%                           for cardiac logfile, 2nd entry for respiratory
%                           logfile
%                           default: 2.5 ms (1/400 Hz)
%       cardiac_modality    'ECG' or 'PULS'/'PPU'/'OXY' to determine
%                           which channel data to be returned
%                           UNUSED, is always pulse plethysmographic unit
%                           for HCP
%       verbose
%       .level              debugging plots are created if level >=3
%       .fig_handles        appended by handle to output figure
%
% OUT
%   cpulse              time events of R-wave peak in cardiac time series (seconds)
%                       <UNUSED>, since not written to logfile
%   r                   respiratory time series
%   t                   vector of time points (in seconds)
%   c                   cardiac time series (PPU)
%   acq_codes           slice/volume start events marked by number <> 0
%                       for time points in t
%                       10/20 = scan start/end;
%                       1 = ECG pulse; 2 = OXY max; 4 = Resp trigger;
%                       8 = scan volume trigger (on)
%                       16 = scan volume trigger (off)
%
% EXAMPLE
%   tapas_physio_read_physlogfiles_siemens_hcp
%
%   See also tapas_physio_read_physlogfiles_siemens tapas_physio_plot_raw_physdata_siemens_hcp

% Author: Lars Kasper
% Created: 2018-01-23
% Copyright (C) 2018 TNU, Institute for Biomedical Engineering,
%                    University of Zurich and ETH Zurich.
%
% This file is part of the TAPAS PhysIO Toolbox, which is released under the terms of the GNU General Public
% License (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

%% read out values
DEBUG = verbose.level >= 2;

hasRespirationFile = ~isempty(log_files.respiration);
hasCardiacFile = ~isempty(log_files.cardiac);

if hasRespirationFile
    y   = load(log_files.respiration, '-ascii');
    trigger_trace_r = y(:,1);
    r   = y(:,2);
else
    r = [];
end

if hasCardiacFile
    y   = load(log_files.cardiac, '-ascii');
    trigger_trace_c = y(:,1);
    c   = y(:,3);
else
    c = [];
end


%% Create timing vector from samples

dt = log_files.sampling_interval;

if isempty(dt)
    dt = 1/400; % 400 Hz sampling interval
end

nSamples = max(numel(c), numel(r));
t = -log_files.relative_start_acquisition + ((0:(nSamples-1))*dt)';


if hasCardiacFile
    trigger_trace = trigger_trace_c;
elseif hasRespirationFile
    trigger_trace = trigger_trace_r;
end

%% Recompute acq_codes as for Siemens (volume on/volume off)
% uses new trigger detection
[acq_codes, verbose] = tapas_physio_create_acq_codes_from_trigger_trace(t, ...
    trigger_trace, verbose);


%% Plot extracted traces so far
if DEBUG
    verbose.fig_handles(end+1) = ...
        tapas_physio_plot_raw_physdata_siemens_hcp(t, c, r, acq_codes);
end

%% Undefined output parameters

cpulse = [];
