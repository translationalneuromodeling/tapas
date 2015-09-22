function [c, r, c_t, c_pulse, verbose] = tapas_physio_read_physlogfiles_siemens(log_files, ...
    verbose, varargin)
% reads out physiological time series and timing vector for Siemens
% logfiles of peripheral cardiac monitoring (ECG/Breathing Belt or
% pulse oximetry)
%
%   [c, r, c_t, c_pulse, verbose] = ...
%       tapas_physio_read_physlogfiles_siemens(log_files, verbose, varargin)
%
% IN    log_files
%       .log_cardiac        contains ECG or pulse oximeter time course
%                           for GE: ECGData...
%       .log_respiration    contains breathing belt amplitude time course
%                           for GE: RespData...
%
% OUT
%   c_cpulse            time events of R-wave peak in cardiac time series (seconds)
%                       for GE: usually empty
%   r                   respiratory time series
%   c_t                 vector of time points (in seconds)
%                       NOTE: This assumes the default sampling rate of 40
%                       Hz
%   c                   cardiac time series (ECG or pulse oximetry)
%
% EXAMPLE
%   [ons_secs.cpulse, ons_secs.rpulse, ons_secs.t, ons_secs.c] =
%       tapas_physio_read_physlogfiles_siemens(logfile, vendor, cardiac_modality);
%
%   See also tapas_physio_main_create_regressors
%
% Author: Lars Kasper
%         file structure information from PhLeM Toolbox, T. Verstynen (November 2007);
%                and Deshpande and J. Grinstead, Siemens Medical Solutions (March 2009)
%         additional log information Miriam Sebold, Charite Berlin (2014)
%
% Created: 2014-07-08
% Copyright (C) 2014 Institute for Biomedical Engineering, ETH/Uni Zurich.
%
% This file is part of the PhysIO toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.
%
% $Id: tapas_physio_read_physlogfiles_siemens_resp.m 781 2015-07-22 16:46:30Z kasperla $

%% read out values

if nargin < 2
    verbose.level = 0;
end
DEBUG = verbose.level >=2;

% process optional input parameters and overwrite defaults
defaults.ecgChannel         = 'mean'; % 'mean'; 'v1'; 'v2'
defaults.endCropSeconds     = 1;

args                = tapas_physio_propval(varargin, defaults);
tapas_physio_strip_fields(args);

dt                  = log_files.sampling_interval;


if ~isempty(log_files.cardiac)
    fid             = fopen(log_files.cardiac);
    s               = dir(log_files.cardiac);
    bufsize         = s.bytes;
% BufSize is needed for older matlab version (comment from Philipp Riedel,
% TU Dresden)
    C               = textscan(fid, '%s', 'Delimiter', '\n', 'BufSize', bufsize);
    fclose(fid);
    
    % Determine relative start of acquisition from dicom headers and
    % logfile footers
    hasScanTimingDicomImage = ~isempty(log_files.scan_timing);
    
    if hasScanTimingDicomImage
        
        %Get time stamps from footer:
        
        linesFooter = C{1}(2:end);
        LogStartTimeSeconds =   str2num(char(regexprep(linesFooter(~cellfun(@isempty,strfind(linesFooter,...
            'LogStartMDHTime'))),'\D',''))) / 1000;
        LogStopTimeSeconds =    str2num(char(regexprep(linesFooter(~cellfun(@isempty,strfind(linesFooter,...
            'LogStopMDHTime'))),'\D',''))) / 1000;
        
        % load dicom
        dicomHeader             = spm_dicom_headers(fullfile(log_files.scan_timing));
        ScanStartTimeSeconds    = dicomHeader{1}.AcquisitionTime;
        ScanStopTimeSeconds     = dicomHeader{1}.AcquisitionTime + ...
            dicomHeader{1}.RepetitionTime/1000;
        
        % This is just a different time-scale, I presume, it does definitely
        % NOT match with the Acquisition time in the DICOM-headers
        % ScanStartTime = str2num(char(regexprep(linesFooter(~cellfun(@isempty,strfind(linesFooter,...
        %     'LogStartMPCUTime'))),'\D','')));
        % ScanStopTime = str2num(char(regexprep(linesFooter(~cellfun(@isempty,strfind(linesFooter,...
        %     'LogStopMPCUTime'))),'\D','')));
        
        switch log_files.align_scan
            case 'first'
                relative_start_acquisition = ScanStartTimeSeconds - ...
                    LogStartTimeSeconds;
            case 'last'
                relative_start_acquisition = ScanStopTimeSeconds - ...
                    LogStopTimeSeconds;
        end
    else
        relative_start_acquisition = 0;
    end           
        
    % add arbitray offset specified by user
    relative_start_acquisition = relative_start_acquisition + ...
        log_files.relative_start_acquisition;
    
    
    c_lineData = C{1}{1};
    c_iTrigger = regexpi(c_lineData, '6002'); % signals start of data logging
    c_lineData = c_lineData((c_iTrigger(end)+4):end);
    c_data = textscan(c_lineData, '%d', 'Delimiter', ' ', 'MultipleDelimsAsOne',1);
    
    % Remove the systems own evaluation of triggers.
    c_pulse  = find(c_data{1} == 5000);  % System uses identifier 5000 as trigger ON
    c_pulse_off = find(c_data{1} == 6000); % System uses identifier 5000 as trigger OFF
    c_recording_on = find(c_data{1} == 6002);% Scanner trigger to Stim PC?
    c_recording_off = find(c_data{1} == 5003);
    
    
    % Filter the trigger markers from the ECG data
     %Note: depending on when the scan ends, the last size(t_off)~=size(t_on).
    c_iNonSignals = [c_pulse; c_pulse_off; c_recording_on; c_recording_off];
    c_codeNonSignals = [5000*ones(size(c_pulse)); ...
        6000*ones(size(c_pulse_off)); ...
        6002*ones(size(c_recording_on))
        5003*ones(size(c_recording_off))];
    
    % data_stream contains only the 2 ECG-channel time courses (with
    % interleaved samples
    c_data_stream = c_data{1};
    c_data_stream(c_iNonSignals) = [];
    
    %iDataStream contains the indices of all true ECG signals in the full
    %data{1}-Array that contains also non-ECG-signals
    c_iDataStream = 1:numel(c_data{1});
    c_iDataStream(c_iNonSignals) = [];
    
    c_nSamples = numel(c_data_stream);
    c_nRows = ceil(c_nSamples/2);
    
    % create a table with channel_1, channels_AVF and trigger signal in
    % different Columns
    % - c_iData_table is a helper table that maps the original indices of the
    % ECG signals in data{1} onto their new positions
    c_data_table = zeros(c_nRows,3);
    c_iData_table = zeros(c_nRows,3);
    
    c_data_table(1:c_nRows,1) = c_data_stream(1:2:end);
    c_iData_table(1:c_nRows,1) = c_iDataStream(1:2:end);
    
    if mod(c_nSamples,2) == 1
        c_data_table(1:c_nRows-1,2) = c_data_stream(2:2:end);
        c_iData_table(1:c_nRows-1,2) = c_iDataStream(2:2:end);
    else
        c_data_table(1:c_nRows,2) = c_data_stream(2:2:end);
        c_iData_table(1:c_nRows,2) = c_iDataStream(2:2:end);
    end
    
    % now fill up 3rd column with trigger data
    % - for each trigger index in data{1}, check where ECG data with closest
    % smaller index ended up in the data_table ... and put trigger code in
    % same row of that table
    c_nTriggers = numel(c_iNonSignals);
    
    for c_iTrigger = 1:c_nTriggers
        % find index before trigger event in data stream and
        % detect it in table
        c_iRow = find(c_iData_table(:,2) == c_iNonSignals(c_iTrigger)-1);
        
        % look in 1st column as well whether maybe signal detected there
        if isempty(c_iRow)
            c_iRow = find(c_iData_table(:,1) == c_iNonSignals(c_iTrigger)-1);
        end
        
        c_data_table(c_iRow,3) = c_codeNonSignals(c_iTrigger);
    end
    
    
    % set new indices to actual
    c_pulse = find(c_data_table(:,3) == 5000);
    c_pulse_off = find(c_data_table(:,3) == 6000);
    c_recording_on = find(c_data_table(:,3) == 6002);
    c_recording_off = find(c_data_table(:,3) == 5003);
    
    % Split a single stream of ECG data into channel 1 and channel 2.
    c_channel_1   = c_data_table(:,1);
    c_channel_AVF = c_data_table(:,2);
    c_meanChannel = mean([c_channel_1(:) c_channel_AVF(:)],2);
    
    % Make them the same length and get time estimates.
    switch ecgChannel
        case 'mean'
            c = c_meanChannel - mean(c_meanChannel);
            
        case 'v1'
            c = c_channel_1 - mean(c_channel_1);
            
        case 'v2'
            c = c_channel_AVF - mean(c_channel_AVF);
    end;
    
    % compute timing vector and time of detected trigger/cpulse events
    c_nSamples = size(c,1);
    c_t = -relative_start_acquisition + ((0:(c_nSamples-1))*dt)';
    c_pulse = c_t(c_pulse);
    c_pulse_off = c_t(c_pulse_off);
    c_recording_on = c_t(c_recording_on);
    c_recording_off = c_t(c_recording_off);
    
    % TODO: put this in log_files.relative_start_acquisition!
    % for now: we assume that log file ends when scan ends (plus a fixed
    % EndClip
    
    c_endClipSamples = floor(endCropSeconds/dt);
    c_stopSample = c_nSamples - c_endClipSamples;
    c_ampl = max(c_meanChannel); % for plotting timing events
    
    if DEBUG
        c_stringTitle = 'Raw Siemens ECG data';
        verbose.fig_handles(end+1) = tapas_physio_get_default_fig_params();
        set(gcf, 'Name', c_stringTitle);
        stem(c_pulse, c_ampl*ones(size(c_pulse)), 'g'); hold all;
        stem(c_pulse_off, c_ampl*ones(size(c_pulse_off)), 'r');
        stem(c_t(c_stopSample), c_ampl , 'm');
        plot(c_t, c_channel_1);
        plot(c_t, c_channel_AVF);
        plot(c_t, c_meanChannel);
       
        c_stringLegend = { ...
            'cpulse on', 'cpulse off', 'assumed last sample of last scan volume', ...
            'channel_1', 'channel_{AVF}', 'mean of channels'};
        
        if ~isempty(c_recording_on)
            stem(c_recording_on, c_ampl*ones(size(c_recording_on)), 'k');
            c_stringLegend{end+1} = 'phys recording on';
        end
        
        if ~isempty(c_recording_off)
            stem(c_recording_off, c_ampl*ones(size(c_recording_off)), 'k');
            c_stringLegend{end+1} = 'phys recording off';
        end
        legend(c_stringLegend);
        title(c_stringTitle);
        xlabel('t (seconds)');
    end
    % crop end of log file
    
    c_pulse(c_pulse > c_t(c_stopSample)) = [];
    c_t(c_stopSample+1:end) = [];
    c(c_stopSample+1:end) = [];
    
else
    c = [];
end

if ~isempty(log_files.respiration)
    fid             = fopen(log_files.respiration);
    s               = dir(log_files.respiration);
    bufsize         = s.bytes;
    % BufSize is needed for older matlab version (comment from Philipp Riedel,
    % TU Dresden)
    R               = textscan(fid, '%s', 'Delimiter', '\n', 'BufSize', bufsize);
    fclose(fid);
        
    
    r_lineData = R{1}{1};
    r_iTrigger = regexpi(r_lineData, '6002'); % signals start of data logging
    r_lineData = r_lineData((r_iTrigger(end)+4):end);
    r_data = textscan(r_lineData, '%d', 'Delimiter', ' ', 'MultipleDelimsAsOne',1);
    
    % Remove the systems own evaluation of triggers.
    r_pulse  = find(r_data{1} == 5000);  % System uses identifier 5000 as trigger ON
    r_pulse_off = find(r_data{1} == 6000); % System uses identifier 5000 as trigger OFF
    r_recording_on = find(r_data{1} == 6002);% Scanner trigger to Stim PC?
    r_recording_off = find(r_data{1} == 5003);
    
    
    % Filter the trigger markers from the Breathing data
     %Note: depending on when the scan ends, the last size(t_off)~=size(t_on).
    r_iNonSignals = [r_pulse; r_pulse_off; r_recording_on; r_recording_off];
    r_codeNonSignals = [5000*ones(size(r_pulse)); ...
        6000*ones(size(r_pulse_off)); ...
        6002*ones(size(r_recording_on))
        5003*ones(size(r_recording_off))];
    
    % data_stream contains one Breathing-channel time courses (with
    % interleaved samples
    r_data_stream = r_data{1};
    r_data_stream(r_iNonSignals) = [];
    
    %iDataStream contains the indices of all true Breath signals in the full
    %data{1}-Array that contains also non-Breath-signals
    r_iDataStream = 1:numel(r_data{1});
    r_iDataStream(r_iNonSignals) = [];
    
    r_nSamples = numel(r_data_stream);
    r_nRows = ceil(r_nSamples/2);
    
    % create a table with channel_1, channels_AVF and trigger signal in
    % different Columns
    % - r_iData_table is a helper table that maps the original indices of the
    % Respiratory signals in data{1} onto their new positions
    r_data_table = zeros(r_nRows,3);
    r_iData_table = zeros(r_nRows,3);
    
    r_data_table(1:r_nRows,1) = r_data_stream(1:2:end);
    r_iData_table(1:r_nRows,1) = r_iDataStream(1:2:end);
    
    if mod(r_nSamples,2) == 1
        r_data_table(1:r_nRows-1,2) = r_data_stream(2:2:end);
        r_iData_table(1:r_nRows-1,2) = r_iDataStream(2:2:end);
    else
        r_data_table(1:r_nRows,2) = r_data_stream(2:2:end);
        r_iData_table(1:r_nRows,2) = r_iDataStream(2:2:end);
    end
    
    % now fill up 3rd column with trigger data
    % - for each trigger index in data{1}, check where Breathing data with closest
    % smaller index ended up in the data_table ... and put trigger code in
    % same row of that table
    r_nTriggers = numel(r_iNonSignals);
    
    for r_iTrigger = 1:r_nTriggers
        % find index before trigger event in data stream and
        % detect it in table
        r_iRow = find(r_iData_table(:,2) == r_iNonSignals(r_iTrigger)-1);
        
        % look in 1st column as well whether maybe signal detected there
        if isempty(r_iRow)
            r_iRow = find(r_iData_table(:,1) == r_iNonSignals(r_iTrigger)-1);
        end
        
        r_data_table(r_iRow,3) = r_codeNonSignals(r_iTrigger);
    end
    
    
    % set new indices to actual
    r_pulse = find(r_data_table(:,3) == 5000);
    r_pulse_off = find(r_data_table(:,3) == 6000);
    r_recording_on = find(r_data_table(:,3) == 6002);
    r_recording_off = find(r_data_table(:,3) == 5003);
    
    % Split a single stream of Respiratory data into channel 1 and channel 2.
    r_channel_1   = r_data_table(:,1);
    r_channel_AVF = r_data_table(:,2);
    r_meanChannel = mean([r_channel_1(:) r_channel_AVF(:)],2);
    
    % Make them the same length and get time estimates.
    switch ecgChannel
        case 'mean'
            r = r_meanChannel - mean(r_meanChannel);
            
        case 'v1'
            r = r_channel_1 - mean(r_channel_1);
            
        case 'v2'
            r = r_channel_AVF - mean(r_channel_AVF);
    end;
    
    % compute timing vector and time of detected trigger/cpulse events
    r_nSamples = size(r,1);
    r_t = -relative_start_acquisition + ((0:(r_nSamples-1))*dt)';
    r_pulse = r_t(r_pulse);
    r_pulse_off = r_t(r_pulse_off);
    r_recording_on = r_t(r_recording_on);
    r_recording_off = r_t(r_recording_off);
    
    % TODO: put this in log_files.relative_start_acquisition!
    % for now: we assume that log file ends when scan ends (plus a fixed
    % EndClip
    
    r_endClipSamples = floor(endCropSeconds/dt);
    r_stopSample = r_nSamples - r_endClipSamples;
    r_ampl = max(r_meanChannel); % for plotting timing events
    
    if DEBUG
        r_stringTitle = 'Raw Siemens Respiratory data';
        verbose.fig_handles(end+1) = tapas_physio_get_default_fig_params();
        set(gcf, 'Name', r_stringTitle);
        stem(r_pulse, r_ampl*ones(size(r_pulse)), 'g'); hold all;
        stem(r_pulse_off, r_ampl*ones(size(r_pulse_off)), 'r');
        stem(r_t(r_stopSample), r_ampl , 'm');
        plot(r_t, r_channel_1);
        plot(r_t, r_channel_AVF);
        plot(r_t, r_meanChannel);
       
        r_stringLegend = { ...
            'cpulse on', 'cpulse off', 'assumed last sample of last scan volume', ...
            'channel_1', 'channel_{AVF}', 'mean of channels'};
        
        if ~isempty(r_recording_on)
            stem(r_recording_on, r_ampl*ones(size(r_recording_on)), 'k');
            r_stringLegend{end+1} = 'phys recording on';
        end
        
        if ~isempty(r_recording_off)
            stem(r_recording_off, r_ampl*ones(size(r_recording_off)), 'k');
            r_stringLegend{end+1} = 'phys recording off';
        end
        legend(r_stringLegend);
        title(r_stringTitle);
        xlabel('t (seconds)');
    end
    % crop end of log file
    
%     r_pulse(r_pulse > r_t(r_stopSample)) = [];
%     r_t(r_stopSample+1:end) = [];
%     r(r_stopSample+1:end) = [];
    
%     r_nSamples = size(r,1);
%     r_t = relative_start_acquisition + ((0:(r_nSamples-1))*dt)';
    
    %     shorten or zero fill r_t to get equal length with c_t;
    r_count = numel(r);
    c_count = numel(c);
    
    if  r_count > c_count
        r(c_count:r_count) = [];
    end
    
    if  r_count < c_count
        r(r_count+1:c_count) = 0;
    end
    

else
    r = [];
end

end
