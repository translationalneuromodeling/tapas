function [c, r, t, cpulse, acq_codes, verbose, gsr] = tapas_physio_read_physlogfiles_adinstruments_txt(...
    log_files, cardiac_modality, verbose, varargin)
% Reads in 4-column txt-export from ADInstruments/LabChart data 
% (e.g., channels titled O2	CO2	Pulse Respiration Volume trigger
% trans_force)
%
% [c, r, t, cpulse, acq_codes, verbose] = tapas_physio_read_physlogfiles_adinstruments_txt(...
%    log_files, cardiac_modality, verbose, varargin)
%
% IN    log_files
%       .log_cardiac        *.txt file, contains header and several columns of the form
%                         ChannelTitle=	Pulse	Respiration	Volume trigger
%                         Range=	20.000 mV	10.000 V	10.000 V
%                         0	8.953	1.245	0.006
%                         0.001	9.609	1.255	0.006
%                         0.002	8.684	1.244	0.007
%                         0.003	9.508	1.250	0.006
%                         0.004	7.490	1.244	0.007
%                         0.005	7.830	1.250	0.006
%                         0.006	3.668	1.239	0.006
%                         0.007	3.283	1.245	0.007
%                         0.008	1.652	1.258	0.007
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
doDebugTriggers = verbose.level >= 3;
doReplaceNans = true;

% if true, assume that trigger switches between +5V and 0 for start of one
% volume, and back to 5V at start of next volume
hasAlternatingTriggerFlank = true; 

hasRespirationFile = ~isempty(log_files.respiration);
hasCardiacFile = ~isempty(log_files.cardiac);

hasRespirationFile = ~isempty(log_files.respiration);
hasCardiacFile = ~isempty(log_files.cardiac);

if hasCardiacFile
    fileName = log_files.cardiac;
elseif hasRespirationFile
    fileName = log_files.respiration;
end


[C, columnNames] = tapas_physio_read_columnar_textfiles(fileName, 'ADINSTRUMENTS_TXT');

% determine right column 
switch lower(cardiac_modality)
    case {'ecg_raw', 'ecg1_raw', 'v1raw'}
        labelColCardiac = 'UNKNOWN'; %TODO: find out!
    case {'oxy','oxyge', 'ppu'}
        labelColCardiac = 'Pulse';
end

labelColResp = 'Respiration';
labelColTrigger = 'Volume trigger';

% it just so works that the first column is just the time (in seconds) per
% row, but labeled "ChannelTitle=", so other column indices line up
% startsWith is needed to ignore trailing end-of-line for column names
if exist('startsWith', 'builtin')
    idxColCardiac = find(startsWith(columnNames, labelColCardiac));
    idxColResp = find(startsWith(columnNames, labelColResp));
    idxColTrigger = find(startsWith(columnNames, labelColTrigger));
else % older Matlab versions, checking chars only up to length of label
    idxColCardiac = find(strncmpi(columnNames, labelColCardiac, numel(labelColCardiac)));
    idxColResp = find(strncmpi(columnNames, labelColResp, numel(labelColResp)));
    idxColTrigger = find(strncmpi(columnNames, labelColTrigger, numel(labelColTrigger)));
end

thresholdTrigger = 4; % Volt, TTL trigger

c = C{idxColCardiac};
r = C{idxColResp};
gsr = C{2}; % TODO: do correctly!
iAcqOn = (C{idxColTrigger}>thresholdTrigger); % trigger is 5V, but flips on/off between volumes

% replace NaNs by max or min depending on nearest non-NaN-neighbour
% (whether it was closer to max or min)
if doReplaceNans
    maxVal = max(c);
    minVal = min(c);
    idxNan = find(isnan(c)); % for loop
    idxValid = find(~isnan(c));

    % find nearest neighbors of valid indices, and replace with min/max,
    % whatever value closest neighbor was closer to
    if exist('knnsearch')
        idxValidClosest = knnsearch(idxValid, idxNan);
        validValClosest = c(idxValid(idxValidClosest));
        isValidValClosestCloserToMin = abs(validValClosest-maxVal) > abs(validValClosest - minVal);
        c(idxNan(isValidValClosestCloserToMin)) = minVal;
        c(idxNan(~isValidValClosestCloserToMin)) = maxVal;
    else % slow...todo: optimize!

        nNans = numel(idxNan);
        iNan = 1;
        while iNan <= nNans

            if ~mod(iNan, 1000)
                fprintf('%d/%d NaNs replaced\n', iNan, nNans);
            end

            idx = idxNan(iNan);

            [~,iValidClosest] = min(abs(idxValid-idx));
            validValClosest = c(idxValid(iValidClosest));

            % choose min or max valid value, whatever is closest
            if abs(maxVal-validValClosest) > abs(validValClosest - minVal)
                c(idx) = minVal;
            else
                c(idx) = maxVal;
            end

            iNan = iNan + 1
        end
    end
end

%% Create timing vector from samples

dt = log_files.sampling_interval;

if isempty(dt)
    dt = mean(diff(C{1})); % first column has timing vector for LabChart
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
    
    % remove duplicate entries
    iAcqStart = unique(iAcqStart);
    iAcqEnd = unique(iAcqEnd);

    if hasAlternatingTriggerFlank
        % choose all rising and falling triggers as volume starts (instead
        % of interpreting falling as an end of a trigger)
        iAcqStartNew = sort([iAcqStart;iAcqEnd]);
        iAcqEndNew = [];
    else
        iAcqStartNew = iAcqStart;
        iAcqEndNew = iAcqEnd;
    end


    if doDebugTriggers
        figure;plot(C{1},C{idxColTrigger})
        hold all;
        stem(C{1}(iAcqStart), C{idxColTrigger}(iAcqStart))
        stem(C{1}(iAcqEnd), C{idxColTrigger}(iAcqEnd))
        stem(C{1}(iAcqStartNew), 0.8*C{idxColTrigger}(iAcqStartNew))
        
        title('Trigger Debugging')
        legend('Volume Trigger Signal', 'Trigger Rising Flank', ...
            'Trigger Falling Flank', 'Chosen Trigger Volume Start')
    end
    
    iAcqStart = iAcqStartNew;
    iAcqEnd = iAcqEndNew;

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
    stringTitle = 'Read-In: Raw ADInstruments/LabChart physlog data (TXT Export)';
    verbose.fig_handles(end+1) = ...
        tapas_physio_plot_raw_physdata_siemens_hcp(t, c, r, acq_codes, ...
        stringTitle);
end

%% Undefined output parameters

cpulse = [];
