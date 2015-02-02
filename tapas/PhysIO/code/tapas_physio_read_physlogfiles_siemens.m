function [c, r, t, cpulse, verbose] = tapas_physio_read_physlogfiles_siemens(log_files, ...
    verbose, varargin)
% reads out physiological time series and timing vector for Siemens
% logfiles of peripheral cardiac monitoring (ECG/Breathing Belt or
% pulse oximetry)
%
%   [cpulse, rpulse, t, c] = tapas_physio_read_physlogfiles_siemens(logfile, vendor, cardiac_modality)
%
% IN    log_files
%       .log_cardiac        contains ECG or pulse oximeter time course
%                           for GE: ECGData...
%       .log_respiration    contains breathing belt amplitude time course
%                           for GE: RespData...
%
% OUT
%   cpulse              time events of R-wave peak in cardiac time series (seconds)
%                       for GE: usually empty
%   r                   respiratory time series
%   t                   vector of time points (in seconds)
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
% $Id: tapas_physio_read_physlogfiles_siemens.m 466 2014-04-27 13:10:48Z kasperla $

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

cpulse              = [];
dt                  = log_files.sampling_interval;


if ~isempty(log_files.cardiac)
    fid             = fopen(log_files.cardiac);
    C               = textscan(fid, '%s', 'Delimiter', '\n');
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
    
    
    lineData = C{1}{1};
    iTrigger = regexpi(lineData, '6002'); % signals start of data logging
    lineData = lineData((iTrigger(end)+4):end);
    data = textscan(lineData, '%d', 'Delimiter', ' ', 'MultipleDelimsAsOne',1);
    
    % Remove the systems own evaluation of triggers.
    cpulse  = find(data{1} == 5000);  % System uses identifier 5000 as trigger ON
    cpulse_off = find(data{1} == 6000); % System uses identifier 5000 as trigger OFF
    recording_on = find(data{1} == 6002);% Scanner trigger to Stim PC?
    recording_off = find(data{1} == 5003);
    
    
    % Filter the trigger markers from the ECG data
     %Note: depending on when the scan ends, the last size(t_off)~=size(t_on).
    iNonEcgSignals = [cpulse; cpulse_off; recording_on; recording_off];
    codeNonEcgSignals = [5000*ones(size(cpulse)); ...
        6000*ones(size(cpulse_off)); ...
        6002*ones(size(recording_on))
        5003*ones(size(recording_off))];
    
    % data_stream contains only the 2 ECG-channel time courses (with
    % interleaved samples
    data_stream = data{1};
    data_stream(iNonEcgSignals) = [];
    
    %iDataStream contains the indices of all true ECG signals in the full
    %data{1}-Array that contains also non-ECG-signals
    iDataStream = 1:numel(data{1});
    iDataStream(iNonEcgSignals) = [];
    
    nSamples = numel(data_stream);
    nRows = ceil(nSamples/2);
    
    % create a table with channel_1, channels_AVF and trigger signal in
    % different Columns
    % - iData_table is a helper table that maps the original indices of the
    % ECG signals in data{1} onto their new positions
    data_table = zeros(nRows,3);
    iData_table = zeros(nRows,3);
    
    data_table(1:nRows,1) = data_stream(1:2:end);
    iData_table(1:nRows,1) = iDataStream(1:2:end);
    
    if mod(nSamples,2) == 1
        data_table(1:nRows-1,2) = data_stream(2:2:end);
        iData_table(1:nRows-1,2) = iDataStream(2:2:end);
    else
        data_table(1:nRows,2) = data_stream(2:2:end);
        iData_table(1:nRows,2) = iDataStream(2:2:end);
    end
    
    % now fill up 3rd column with trigger data
    % - for each trigger index in data{1}, check where ECG data with closest
    % smaller index ended up in the data_table ... and put trigger code in
    % same row of that table
    nTriggers = numel(iNonEcgSignals);
    
    for iTrigger = 1:nTriggers
        % find index before trigger event in data stream and
        % detect it in table
        iRow = find(iData_table(:,2) == iNonEcgSignals(iTrigger)-1);
        
        % look in 1st column as well whether maybe signal detected there
        if isempty(iRow)
            iRow = find(iData_table(:,1) == iNonEcgSignals(iTrigger)-1);
        end
        
        data_table(iRow,3) = codeNonEcgSignals(iTrigger);
    end
    
    
    % set new indices to actual
    cpulse = find(data_table(:,3) == 5000);
    cpulse_off = find(data_table(:,3) == 6000);
    recording_on = find(data_table(:,3) == 6002);
    recording_off = find(data_table(:,3) == 5003);
    
    % Split a single stream of ECG data into channel 1 and channel 2.
    channel_1   = data_table(:,1);
    channel_AVF = data_table(:,2);
    meanChannel = mean([channel_1(:) channel_AVF(:)],2);
    
    % Make them the same length and get time estimates.
    switch ecgChannel
        case 'mean'
            c = meanChannel - mean(meanChannel);
            
        case 'v1'
            c = channel_1 - mean(channel_1);
            
        case 'v2'
            c = channel_AVF - mean(channel_AVF);
    end;
    
    % compute timing vector and time of detected trigger/cpulse events
    nSamples = size(c,1);
    t = -relative_start_acquisition + ((0:(nSamples-1))*dt)';
    cpulse = t(cpulse);
    cpulse_off = t(cpulse_off);
    recording_on = t(recording_on);
    recording_off = t(recording_off);
    
    % TODO: put this in log_files.relative_start_acquisition!
    % for now: we assume that log file ends when scan ends (plus a fixed
    % EndClip
    
    endClipSamples = floor(endCropSeconds/dt);
    stopSample = nSamples - endClipSamples;
    ampl = max(meanChannel); % for plotting timing events
    
    if DEBUG
        stringTitle = 'Raw Siemens physlog data';
        verbose.fig_handles(end+1) = tapas_physio_get_default_fig_params();
        set(gcf, 'Name', stringTitle);
        stem(cpulse, ampl*ones(size(cpulse)), 'g'); hold all;
        stem(cpulse_off, ampl*ones(size(cpulse_off)), 'r');
        stem(t(stopSample), ampl , 'm');
        plot(t, channel_1);
        plot(t, channel_AVF);
        plot(t, meanChannel);
       
        stringLegend = { ...
            'cpulse on', 'cpulse off', 'assumed last sample of last scan volume', ...
            'channel_1', 'channel_{AVF}', 'mean of channels'};
        
        if ~isempty(recording_on)
            stem(recording_on, ampl*ones(size(recording_on)), 'k');
            stringLegend{end+1} = 'phys recording on';
        end
        
        if ~isempty(recording_off)
            stem(recording_off, ampl*ones(size(recording_off)), 'k');
            stringLegend{end+1} = 'phys recording off';
        end
        legend(stringLegend);
        title(stringTitle);
        xlabel('t (seconds)');
    end
    % crop end of log file
    
    cpulse(cpulse > t(stopSample)) = [];
    t(stopSample+1:end) = [];
    c(stopSample+1:end) = [];
    
else
    c = [];
end

if ~isempty(log_files.respiration)
    r = load(log_files.respiration, 'ascii');
    nSamples = size(r,1);
    t = relative_start_acquisition + ((0:(nSamples-1))*dt)';
else
    r = [];
end

end
