function [data_table, log_parts] = tapas_physio_siemens_line2table(lineData, cardiacModality)
% transforms data line of Siemens log file (before linebreak after 5003
% signal for recording end) into table (sorting amplitude and trigger signals)
%
%   data_table = tapas_physio_siemens_line2table(input)
%
% IN
%   lineData        all recording amplitude samples are saved in first line
%                   of file. See also tapas_physio_read_physlogfiles_siemens_raw
%
% OUT
%   data_table      [nSamples,nChannels+1] table of recording channel_1, ..., channel_N and trigger
%                   signal with trigger codes:
%                   5000 = cardiac pulse on
%                   6000 = cardiac pulse off
%                   6002 = phys recording on
%                   6003 = phys recording off
%   log_parts       part of logfile according to markers by Siemens
%
% EXAMPLE
%   tapas_physio_siemens_line2table
%
%   See also

% Author: Lars Kasper
% Created: 2016-02-29
% Copyright (C) 2016 TNU, Institute for Biomedical Engineering,
%                    University of Zurich and ETH Zurich.
%
% This file is part of the TAPAS PhysIO Toolbox, which is released under the terms of the GNU General Public
% License (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.


% NOTE: The following extraction of data from the logfiles might seem
% parsimonious, but it's written in a way to support all different physio
% trace modalities and logfile versions (more detail below)

% The typical logfile structure is as follows (all data in first line of
% logfile, the footer is in the next line (after 5003), not used in this
% file, but tapas_physio_read_physlogfiles_siemens_raw)
%
% <Header> 5002 <LOGVERSION XX> 6002
% <[optional] training trace data> 5002 uiHwRevisionPeru ... [optional] 6002
% 5002 <infoRegion3> 6002 5002 <infoRegion4> 6002 ... 5002 <infoRegionN> 6002
% <trace data 1 (all channels, arbitrary number of samples, trigger markers 5000, 6000)> ...
% 5002 <infoRegionN+1> 6002
% <trace data 2 (all channels, arbitrary number of samples, trigger markers 5000, 6000)> ...
% 5002 <infoRegionN+2> 6002 ...
% <trace data M> ... 5003
%
% Since the number and content of actual info regions and number of data
% channels differs by trace modality and logfile version, we have to cut
% data out carefully:
%
% 1. Cut away header
% 2. Cut away logfile version
% 3. Cut away optional training trace
% 4. Remove all infoRegions can be interleaved with trace data, e.g.,
%    cushionGain for RESP trace)
% 5. Remove all trigger markers (5000, 6000), but remmember position
% 6. Sort remaining trace into coresponding channels (number of channels is
% logfile-version dependent)
% 7. Re-insert trigger markers as extra column



%% 1. Header goes from start of line to first occurence of 5002
[iStartHeader, iEndHeader, logHeader] = regexp(lineData, '^(.*?) (?=\<5002\>)', 'start', 'end', 'tokens' );
logHeader = logHeader{1}{1};
lineData(iStartHeader:iEndHeader) = []; % remove header, but keep 5002 for next step


%% 2. Logfile version (which alters no of channels etc.)
% stored in first 5002/6002 info region
%   5002 LOGVERSION   1 6002%
%   5002 LOGVERSION_RESP   3 6002%
%   5002 LOGVERSION_PULS   3 6002%
[iStartVersionInfo, iEndVersionInfo, logVersion] = regexp(lineData, '^\<5002\> LOGVERSION[^0-9]*(\d+)\s\<6002\>', 'start', 'end', 'tokens' );
logVersion = str2double(logVersion{1}{1});
lineData(iStartVersionInfo:iEndVersionInfo) = []; % remove version info region incl. 5002/6002 markers


%% 3. Optional training data (for Siemens own peak detection) is after
% "6002" of Logversion info and "5002 uiHwRevisionPeru"
[iStartTraining, iEndTraining, dataTraining] = regexp(lineData, '^\s*(.*?) (?=\<5002\> uiHwRevisionPeru)', 'start', 'end', 'tokens');
if ~isempty(iStartTraining) % training trace does not always exist
    dataTraining = dataTraining{1}{1};
    lineData(iStartTraining:iEndTraining) = []; % remove training trace, but keep following 5002 for next step
end


%% 4. Identify and remove info regions between 5002 and 6002 (may be
% interleaved with trace data (e.g., messages or cushion Gain for RESP)
[iStartInfoRegion, iEndInfoRegion, logInfoRegion] = regexp(lineData, '\<5002\>(.*?)\<6002\>', 'start', 'end', 'tokens' );
logInfoRegion = cellfun(@(x) x{1,1}, logInfoRegion, 'UniformOutput', false)';
traceData = regexprep(lineData, '\<5002\>(.*?)\<6002\>\s', '');
traceData = regexprep(traceData, '\<5003\>$', ''); % remove 5003 mark of trace end

log_parts.logHeader = logHeader;
log_parts.logInfoRegion = logInfoRegion;
log_parts.logVersion = logVersion;

% convert remaining data (all numbers string) to number (int32)
data = textscan(traceData, '%d', 'Delimiter', ' ', 'MultipleDelimsAsOne', true);


%% 5. Remove all trigger markers (5000, 6000), but remmember position
% Remove the systems own evaluation of triggers.
cpulse  = find(data{1} == 5000);  % System uses identifier 5000 as trigger ON
cpulse_off = find(data{1} == 6000); % System uses identifier 5000 as trigger OFF
% Filter the trigger markers from the ECG data
% Note: depending on when the scan ends, the last size(t_off)~=size(t_on).
iTriggerMarker = [cpulse; cpulse_off];
codeTriggerMarker = [5000*ones(size(cpulse)); ...
    6000*ones(size(cpulse_off))];

% data_stream contains only the time courses (with
% interleaved samples for each channel)
data_stream = data{1};
data_stream(iTriggerMarker) = [];

%iDataStream contains the indices of all true trace signals in the full
%data{1}-Array that contains also the trigger markers
iDataStream = 1:numel(data{1});
iDataStream(iTriggerMarker) = [];


%% 6. Sort remaining trace into coresponding channels (number of channels is
% logfile-version dependent)
nSamples = numel(data_stream);
switch upper(cardiacModality) % ecg has two channels, resp and puls only one
    case 'ECG'
        switch logVersion
            case 1
                nChannels = 2;
            case 3
                nChannels = 4;
        end
    case 'PPU'
        nChannels = 1;
    case 'RESP'
        switch logVersion
            case 1
                nChannels = 1;
            case 3
                nChannels = 5; % breathing belt plus 4 channel biomatrix
        end
        
    otherwise
        error('unknown cardiac/respiratory logging modality: %s', cardiacModality);
end

nRows = ceil(nSamples/nChannels);

% create a table with channel_1, channels_AVF and trigger signal in
% different Columns
% - iData_table is a helper table that maps the original indices of the
% ECG signals in data{1} onto their new positions
data_table = zeros(nRows,nChannels+1);
iData_table = zeros(nRows,nChannels+1);

for iChannel = 1:nChannels
    data_table(1:nRows,iChannel) = data_stream(iChannel:nChannels:end);
    iData_table(1:nRows,iChannel) = iDataStream(iChannel:nChannels:end);
end

% TODO: deal with mod(nSamples, nChannels) > 0 (incomplete data?)


%% 7. Re-insert trigger markers as extra column
% now fill up nChannel+1. column with trigger data
% - for each trigger index in data{1}, check where ECG data with closest
% smaller index ended up in the data_table ... and put trigger code in
% same row of that table
nTriggers = numel(iTriggerMarker);

for iTrigger = 1:nTriggers
    % find index before trigger event in data stream and
    % detect it in table, look in last columns first, then go
    % backwards
    iRow = [];
    iChannel = nChannels;
    while isempty(iRow)
        
        iRow = find(iData_table(:,iChannel) == iTriggerMarker(iTrigger)-1);
        iChannel = iChannel - 1;
    end
    
    data_table(iRow,nChannels+1) = codeTriggerMarker(iTrigger);
end