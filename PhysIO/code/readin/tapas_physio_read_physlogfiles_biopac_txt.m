function [c, r, t, cpulse, acq_codes, verbose, gsr] = tapas_physio_read_physlogfiles_biopac_txt(...
    log_files, cardiac_modality, verbose, varargin)
% Reads in 4-column txt-export from BioPac Data (resp, ppu, gsr, trigger)
%
% [c, r, t, cpulse, acq_codes, verbose] = tapas_physio_read_physlogfiles_biopac_txt(...
%    log_files, cardiac_modality, verbose, varargin)
%
% IN    log_files
%       .log_cardiac        *.txt file, contains 4 columns of the form
%                           RESP - RSP100C	GSR - EDA100C-MRI	PPG - PPG100C	Marker
%                             -0.949402	-0.00610382	0.0134277	0
%                             -0.949402	-0.00610382	0.0134277	0
%                             -0.951233	-0.00915558	0.0204468	0
%                             -0.951233	-0.00915558	0.0204468	0
%                             -0.953064	-0.0122073	0.0259399	0
%                             -0.953064	-0.0122073	0.0259399	0
%                             -0.95459	-0.0076297	0.0296021	0
%                             -0.95459	-0.0076297	0.0296021	0
%       .log_respiration    same as .log_cardiac
%       .sampling_interval  sampling interval (in seconds)
%                           default: 1 ms (1000 Hz)
%       cardiac_modality    'ECG' or 'PULS'/'PPU'/'OXY' to determine
%                           which channel data to be returned
%                           UNUSED, is always pulse plethysmographic unit
%                           for BioPac
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
%   gsr                 galvanic skin response (not used)
%
% EXAMPLE
%   tapas_physio_read_physlogfiles_biopac_txt
%
%   See also tapas_physio_read_physlogfiles_siemens tapas_physio_plot_raw_physdata_siemens_hcp

% Author: Lars Kasper
% Created: 2018-09-27
% Copyright (C) 2018 TNU, Institute for Biomedical Engineering,
%                    University of Zurich and ETH Zurich.

% This file is part of the TAPAS PhysIO Toolbox, which is released under the terms of the GNU General Public
% License (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

%% read out values
DEBUG = verbose.level >= 2;

hasRespirationFile = ~isempty(log_files.respiration);
hasCardiacFile = ~isempty(log_files.cardiac);

hasRespirationFile = ~isempty(log_files.respiration);
hasCardiacFile = ~isempty(log_files.cardiac);

if hasCardiacFile
    fileName = log_files.cardiac;
elseif hasRespirationFile
    fileName = log_files.respiration;
end


[C, columnNames] = tapas_physio_read_columnar_textfiles(fileName, 'BIOPAC_TXT');
c = double(C{3});
r = double(C{1});
gsr = double(C{2});
iAcqOn = (double(C{4})~=0); % trigger has 11, rest is 0;

%% Create timing vector from samples

dt = log_files.sampling_interval;

if isempty(dt)
    dt = 1/1000; % 1000 Hz sampling interval
end

nSamples = max(numel(c), numel(r));
t = -log_files.relative_start_acquisition + ((0:(nSamples-1))*dt)';

%% Recompute acq_codes as for Siemens (volume on/volume off)
acq_codes = [];

if ~isempty(iAcqOn) % otherwise, nothing to read ...
    % iAcqOn is a column of 1s and 0s, 1 whenever scan acquisition is on
    % Determine 1st start and last stop directly via first/last 1
    % Determine everything else in between via difference (go 1->0 or 0->1)
    iAcqStart   = find(iAcqOn, 1, 'first');
    iAcqEnd     = find(iAcqOn, 1, 'last');
    d_iAcqOn    = diff(iAcqOn);
    
    % index shift + 1, since diff vector has index of differences i_(n+1) - i_n,
    % and the latter of the two operands (i_(n+1)) has sought value +1
    iAcqStart   = [iAcqStart; find(d_iAcqOn == 1) + 1];
    % no index shift, for the same reason
    iAcqEnd     = [find(d_iAcqOn == -1); iAcqEnd];
    
    acq_codes = zeros(nSamples,1);
    acq_codes(iAcqStart) = 8; % to match Philips etc. format
    acq_codes(iAcqEnd) = 16; % don't know...
    
    % report estimated onset gap between last slice of volume_n and 1st slice of
    % volume_(n+1)
    nAcqStarts = numel(iAcqStart);
    nAcqEnds = numel(iAcqEnd);
    nAcqs = min(nAcqStarts, nAcqEnds);
    
    if nAcqs >= 1
        % report time of acquisition, as defined in SPM
        TA = mean(t(iAcqEnd(1:nAcqs)) - t(iAcqStart(1:nAcqs)));
        verbose = tapas_physio_log(...
            sprintf('TA = %.4f s (Estimated time of acquisition during one volume TR)', ...
            TA), verbose, 0);
    end
end


%% Plot, if wanted

if DEBUG
    stringTitle = 'Read-In: Raw BioPac physlog data (TXT Export)';
    verbose.fig_handles(end+1) = ...
        tapas_physio_plot_raw_physdata_siemens_hcp(t, c, r, acq_codes, ...
        stringTitle);
end

%% Undefined output parameters

cpulse = [];
