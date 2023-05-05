function dataCardiac = tapas_physio_siemens_table2cardiac(data_table, ...
    ecgChannel, dt, relative_start_acquisition, endCropSeconds)
% extract structured data from table of channel signal and trigger events
%
%      dataCardiac = tapas_physio_table2cardiac(...
%           data_table, ecgChannel, relative_start_acquisition, endCropSeconds);
%
% IN
%   data_table      [nSamples,3] table of channel_1, channels_AVF and trigger 
%                   signal with trigger codes:
%                   5000 = cardiac pulse on
%                   6000 = cardiac pulse off
%                   6002 = phys recording on
%                   6003 = phys recording off
%   ecgChannel      'v1', 'v2', or 'mean'
%   relative_start_acquisition 
%                   start of logfile relative to
%                   onset of first scan (t=0)
%
% OUT
%
% dataCardiac = struct(...
%     'cpulse_on', cpulse_on, ...
%     'cpulse_off', cpulse_off, ...
%     'recording_on', recording_on, ...
%     'recording_off', recording_off, ...
%     'recordingChannels', recordingChannels, ...
%     'meanChannel', meanChannel, ...
%     'c', c, ...
%     't', t, ...
%     'ampl', ampl, ...
%     'stopSample', stopSample ...
%     );
%   
%   AVF means "Augmented Vector Foot" and is a label for a unipolar lead
%   of the ECG electrode setup. For .ecg- LOGVERSIOM one, this was the 2nd
%   column ("channel") in the data (based on Siemens' Physioload function), 
%   but for later versions, it's not clear
%
% EXAMPLE
%   tapas_physio_siemens_table2cardiac
%
%   See also tapas_physio_siemens_line2table
%   See also tapas_physio_read_physlogfiles_siemens

% Author: Lars Kasper
% Created: 2016-02-29
% Copyright (C) 2016 TNU, Institute for Biomedical Engineering,
%                    University of Zurich and ETH Zurich.
%
% This file is part of the TAPAS PhysIO Toolbox, which is released under the terms of the GNU General Public
% License (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.


% set new indices to actual, last column contains 
cpulse_on          = find(data_table(:,end) == 5000);
cpulse_off      = find(data_table(:,end) == 6000);
recording_on    = find(data_table(:,end) == 6002);
recording_off   = find(data_table(:,end) == 5003);

% Split a single stream of ECG data into channel 1 and channel 2.
nChannels = size(data_table,2) - 1;

for iChannel = 1:nChannels
   recordingChannels(:,iChannel) = data_table(:,iChannel); 
end

meanChannel = mean(recordingChannels, 2);

% Make them the same length and get time estimates.
switch lower(ecgChannel)
    case 'mean'
        c = meanChannel - mean(meanChannel);
        
    case {'v1', 'v2', 'v3', 'v4'}
        iChannel = str2double(ecgChannel(2));
        c = recordingChannels(:,iChannel) - mean(recordingChannels(:,iChannel));
    case {'avf'}
        iChannel =2;
        c = recordingChannels(:,iChannel) - mean(recordingChannels(:,iChannel));
end

% compute timing vector and time of detected trigger/cpulse events
nSamples = size(c,1);
t = -relative_start_acquisition + ((0:(nSamples-1))*dt)';

cpulse_on = t(cpulse_on);
cpulse_off = t(cpulse_off);
recording_on = t(recording_on);
recording_off = t(recording_off);

% TODO: put this in log_files.relative_start_acquisition!
% for now: we assume that log file ends when scan ends (plus a fixed
% EndClip

endClipSamples = floor(endCropSeconds/dt);
stopSample = nSamples - endClipSamples;
ampl = max(meanChannel); % for plotting timing events

dataCardiac = struct(...
    'cpulse_on', cpulse_on, ...
    'cpulse_off', cpulse_off, ...
    'recording_on', recording_on, ...
    'recording_off', recording_off, ...
    'recordingChannels', recordingChannels, ...
    'meanChannel', meanChannel, ...
    'c', c, ...
    't', t, ...
    'ampl', ampl, ...
    'stopSample', stopSample ...
    );