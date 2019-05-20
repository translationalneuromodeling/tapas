function [c, r, t, cpulse, acq_codes, verbose] = tapas_physio_read_physlogfiles_siemens_tics(...
    log_files, cardiac_modality, verbose, varargin)
% reads out physiological time series of Siemens logfiles with Tics
% The latest implementation of physiological logging in Siemens uses Tics,
% i.e. time stamps of 2.5 ms duration that are reset every day at midnight.
% These are used as a common time scale in all physiological logfiles -
% even though individual sampling times may vary - including cardiac,
% respiratory, pulse oximetry and acquisition time data itself
%
% [c, r, t, cpulse, acq_codes, verbose] = tapas_physio_read_physlogfiles_siemens_tics(...
%    log_files, cardiac_modality, verbose)
%
% IN    log_files
%       .log_cardiac        contains ECG or pulse oximeter time course
%                           for Siemens: *_PULS.log or _ECG[1-4].log.
%       .log_respiration    contains breathing belt amplitude time course
%                           for Siemens: *_RESP.log
%       .sampling_interval  1 entry: sampling interval (seconds)
%                           for both log files
%                           2 entries: 1st entry sampling interval (seconds)
%                           for cardiac logfile, 2nd entry for respiratory
%                           logfile
%                           default: 2.5 ms (1/400 Hz)
%       cardiac_modality    'ECG' or 'PULS'/'PPU'/'OXY' to determine
%                           which channel data to be returned
%                           if not given, will be read out from file name
%                           suffix
%       verbose
%       .level              debugging plots are created if level >=3
%       .fig_handles        appended by handle to output figure
%
%       varargin            propertyName/value pairs, as follows
%           'ecgChannel'    'v1', 'v2', 'v3', 'v4', 'mean' (default)
%                           determines which ECG channel to use as
%                           output cardiac curve
% OUT
%   cpulse              time events of R-wave peak in cardiac time series (seconds)
%   r                   respiratory time series
%   t                   vector of time points (in seconds)
%   c                   cardiac time series (ECG or pulse oximetry)
%   acq_codes           slice/volume start events marked by number <> 0
%                       for time points in t
%                       10/20 = scan start/end;
%                       1 = ECG pulse; 2 = OXY max; 4 = Resp trigger;
%                       8 = scan volume trigger
%
% EXAMPLE
%   [ons_secs.cpulse, ons_secs.rpulse, ons_secs.t, ons_secs.c] =
%       tapas_physio_read_physlogfiles_siemens_tics(logfiles);
%
%   See also tapas_physio_main_create_regressors

% Author: Lars Kasper
% Created: 2014-09-08
% Copyright (C) 2014 Institute for Biomedical Engineering, ETH/Uni Zurich.
%
% This file is part of the PhysIO toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.


%% read out values
DEBUG = verbose.level >= 3;

% channel selection different in ECG data, multiple lines for multiple
% channel, labeled in log file
switch upper(cardiac_modality)
    case 'ECG'
        defaults.ecgChannel = 'v1'; % 'mean'; 'v1'; 'v2', 'v3', 'v4'
    case {'PPU', 'OXY', 'PMU', 'PULS'}
        defaults.ecgChannel = 'PULS';
end

args = tapas_physio_propval(varargin, defaults);
tapas_physio_strip_fields(args);


hasRespirationFile = ~isempty(log_files.respiration);
hasCardiacFile = ~isempty(log_files.cardiac);

% Cardiac and respiratory sampling intervals are ignored, since Tics are
% assumed to be counted in files
if ~isempty(log_files.sampling_interval)
    verbose = tapas_physio_log( ...
        'Ignoring given sampling intervals, using dt = 2.5 ms (tics) instead', ...
        verbose, 1);
end

dtTics = 2.5e-3;
dtCardiac = dtTics;
dtRespiration = dtTics;

%% Define outputs
t           = [];
acq_codes   = [];
tCardiac    = [];
cacq_codes  = [];
c           = [];
cpulse      = [];
tRespiration = [];
racq_codes  = [];
r           = [];
rpulse      = [];

%% Read resp. file
if hasRespirationFile
    
    C = tapas_physio_read_columnar_textfiles(log_files.respiration, 'RESP');
    nColumns = numel(C); % different file formats indicated by diff number of columns
    
    extTriggerSignals = [];
    switch nColumns
        case 3 % Cologne format
            r           = double(C{2});
            rSignals    = double(C{3});
        case {4,5}
            r           = double(C{3});
            rSignals    = ~cellfun(@isempty, C{4});
    end
    
    if nColumns == 5
        extTriggerSignals = ~cellfun(@isempty, C{5});
    end
    
    rTics           = double(C{1});
    tRespiration    = rTics*dtRespiration ...
        - log_files.relative_start_acquisition;
    
    rpulse          = find(rSignals);
    
    nSamples        = numel(C{1});
    racq_codes       = zeros(nSamples,1);
    
    if ~isempty(rpulse)
        racq_codes(rpulse) = racq_codes(rpulse) + 4;
        rpulse = tRespiration(rpulse);
    end
    
    acqpulse          = find(extTriggerSignals);
    
    if ~isempty(acqpulse)
        racq_codes(acqpulse) = racq_codes(acqpulse) + 8;
    end
end

%% Read cardiac file
if hasCardiacFile
    
    C = tapas_physio_read_columnar_textfiles(log_files.cardiac);
    
    nColumns = numel(C);
    
    % different file formats indicated by diff number of columns
    
    extTriggerSignals = [];
    switch nColumns
        case 3 % Cologne format
            c           = double(C{2});
            cSignals    = double(C{3});
            cTics           = double(C{1});
            
        case {4,5}
            [cTics, c, cSignals, extTriggerSignals, stringChannels, verbose] = ...
                tapas_physio_split_data_per_channel_siemens_tics(C, ecgChannel, verbose);
    end
    
    
    tCardiac        = cTics*dtCardiac ...
        - log_files.relative_start_acquisition;
    
    nSamples         = numel(c);
    cacq_codes       = zeros(nSamples,1);
    
    cpulse          = find(cSignals);
    
    if ~isempty(cpulse)
        isOxy = any(strfind(upper(log_files.cardiac), 'PULS')); % different codes for PPU
        
        cacq_codes(cpulse) = cacq_codes(cpulse) + 1 + isOxy; %+1 for ECG, +2 for PULS
        cpulse = tCardiac(cpulse);
    end
    
    acqpulse          = find(extTriggerSignals);
    
    if ~isempty(acqpulse)
        cacq_codes(acqpulse) = cacq_codes(acqpulse) + 8;
    end
end



%% plot raw data so far
if DEBUG
    fh = tapas_physio_plot_raw_physdata_siemens_tics(tCardiac, c, tRespiration, r, ...
        hasCardiacFile, hasRespirationFile, cpulse, rpulse, cacq_codes, ...
        racq_codes);
    verbose.fig_handles(end+1) = fh;
end


%% If only one file exists, take t and acq_codes from that accordingly
hasOnlyCardiacFile = hasCardiacFile && ~hasRespirationFile;
if hasOnlyCardiacFile
    t = tCardiac;
    acq_codes = cacq_codes;
end

hasOnlyRespirationFile = ~hasCardiacFile && hasRespirationFile;
if hasOnlyRespirationFile
    t = tRespiration;
    acq_codes = racq_codes;
end

%% Merge acquisition codes, if both files exist, but on same sampling grid
if hasCardiacFile && hasRespirationFile
    
    haveSameSampling = isequal(tRespiration, tCardiac);
    if haveSameSampling
        acq_codes = cacq_codes + racq_codes;
    else
        %% interpolate to greater precision, if both files exist and
        % 2 different sampling rates are given
        %interpolate acq_codes and trace with lower sampling rate to higher
        %rate
        
        dtCardiac = tCardiac(2)-tCardiac(1);
        dtRespiration = tRespiration(2) - tRespiration(1);
        
        isHigherSamplingCardiac = dtCardiac < dtRespiration;
        if isHigherSamplingCardiac
            t = tCardiac;
            rInterp = interp1(tRespiration, r, t);
            racq_codesInterp = interp1(tRespiration, racq_codes, t, 'nearest');
            acq_codes = cacq_codes + racq_codesInterp;
            
            if DEBUG
                fh = plot_interpolation(tRespiration, r, t, rInterp, ...
                    {'respiratory', 'cardiac'});
                verbose.fig_handles(end+1) = fh;
            end
            r = rInterp;
            
        else
            t = tRespiration;
            cInterp = interp1(tCardiac, c, t);
            cacq_codesInterp = interp1(tCardiac, cacq_codes, t, 'nearest');
            acq_codes = racq_codes + cacq_codesInterp;
            
            if DEBUG
                fh = plot_interpolation(tCardiac, c, t, cInterp, ...
                    {'cardiac', 'respiratory'});
                verbose.fig_handles(end+1) = fh;
            end
            c = cInterp;
            
        end
    end
end
end

%% Local function to plot interpolation result
function fh = plot_interpolation(tOrig, yOrig, tInterp, yInterp, ...
    stringOrigInterp)
fh = tapas_physio_get_default_fig_params();
stringTitle = sprintf('Read-In: Interpolation of %s signal', stringOrigInterp{1});
set(fh, 'Name', stringTitle);
plot(tOrig, yOrig, 'go--');  hold all;
plot(tInterp, yInterp,'r.');
legend({
    sprintf('after interpolation to %s timing', ...
    stringOrigInterp{1}), ...
    sprintf('original %s time series', stringOrigInterp{2}) });
title(stringTitle);
xlabel('time (seconds');
end

