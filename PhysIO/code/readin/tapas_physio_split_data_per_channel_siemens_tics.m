function [cTics, c, cSignals, extTriggerSignals, stringChannels, verbose, dtTics] = ...
    tapas_physio_split_data_per_channel_siemens_tics(C, ecgChannel, verbose)
% Splits column data of Siemens Tics format according to their channel
% label in the 2nd column
%
%      [cTics, c, cSignals, extTriggerSignals, stringChannels] = ...
%                tapas_physio_split_data_per_channel_siemens_tics(C, ecgChannel);
%
%
%
%   Example structure of a tics logfile (CMRR):
%
%      19824786     ECG2   2084
%      19824787     ECG2   2190
%      19824788     ECG2   2198  PULS_TRIGGER
%      19824789     ECG2   2095
%      ...
%      19824762     ECG3   1948
%      19824763     ECG3   1940
%      19824764     ECG3   1953
%
% IN
%   ecgChannel      string indicating which channel (combination) shall be
%                   used
%                   'c1' (='PULS'), 'c2', 'c3', 'c4', ...
%                       single channel data is returned
%                   'v12'(='v1'), 'v23', .... v34 (='v2')
%                       voltage difference channel 1 - 2, 2-3 etc. is
%                       returned
%                   'mean'  = 1/2 (v12 + v34);
% OUT
%
% EXAMPLE
%   tapas_physio_split_data_per_channel_siemens_tics
%
%   See also

% Author: Lars Kasper
% Created: 2017-11-17
% Copyright (C) 2017 TNU, Institute for Biomedical Engineering,
%                    University of Zurich and ETH Zurich.
%
% This file is part of the TAPAS PhysIO Toolbox, which is released under the terms of the GNU General Public
% License (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.



DEBUG = verbose.level >= 3;

if nargin < 4
    dtTics = 2.5e-3;
end

c           = double(C{3});
cSignals    = ~cellfun(@isempty, C{4});


stringChannels  = unique(C{2});
nChannels       = numel(stringChannels);
nColumns        = numel(C);

cTics           = double(C{1});

if nColumns == 5
    extTriggerSignals = ~cellfun(@isempty, C{5});
else
    extTriggerSignals = zeros(numel(c),1);
end


cTicsUnique = unique(cTics);

%% retrieve data channel-wise, and interpolate to common grid
for iChannel = 1:nChannels
    idxChannel = find(strcmp(C{2}, stringChannels{iChannel}));
    cChannel = c(idxChannel);
    cTicsChannel = cTics(idxChannel);
    cSignalsChannel = cSignals(idxChannel);
    extTriggerSignalsChannel = extTriggerSignals(idxChannel);
    
    %% interpolate to same time grid (tics) for channel combination  already...
    
    % first, remove duplicates in tics time axis by averaging
    % their corresponding values, keep only one occurence
    
    % detects bins with more than one entry, index of first
    % occurence is returned
    idxDuplicateTics = find(hist(cTicsChannel,unique(cTicsChannel))>1);
    
    idxDuplicatesAll = [];
    for idx = idxDuplicateTics
        % detect all occurences that match first labeled as
        % duplicate
        idxDuplicates = reshape(find(cTicsChannel==cTicsChannel(idx)), 1, []);
        cChannel(idxDuplicates(1)) = mean(cChannel(idxDuplicateTics));
        
        % label for later deletion without changing index order
        % just now
        idxDuplicatesAll = [idxDuplicatesAll, idxDuplicates(2:end)];
    end
    
    % remove duplicates
    cChannel(idxDuplicatesAll)                      = [];
    cTicsChannel(idxDuplicatesAll)                  = [];
    idxChannel(idxDuplicatesAll)                    = [];
    cSignalsChannel(idxDuplicatesAll)               = [];
    extTriggerSignalsChannel(idxDuplicatesAll)      = [];
    
    % now interpolate without duplicates!
    cChannelInterpolated = interp1(cTicsChannel, cChannel, ...
        cTicsUnique, 'linear', 'extrap');
    
    % save non-duplicate and interpolated data for channel
    indPerChannel{iChannel}                         = idxChannel;
    cPerChannel{iChannel}                           = cChannel;
    cTicsPerChannel{iChannel}                       = cTicsChannel;
    cPerChannelInterpolated{iChannel}               = cChannelInterpolated;
    cSignalsPerChannel{iChannel}                    = cSignalsChannel;
    extTriggerSignalsPerChannel{iChannel}    = extTriggerSignalsChannel;
end

cTics = cTicsUnique;

% alter ecgChannel-selection if invalid (too small) number of channels
% available
if nChannels == 1 && ~ismember(lower(ecgChannel), {'c1', 'puls'})
    ecgChannel = 'c1';
    warning('Changed selected ECG channel to channel 1');
end

if nChannels < 4 && ~ismember(lower(ecgChannel), {'c1', 'c2', 'puls', 'v12'})
    ecgChannel = 'v12';
    warning('Changed selected ECG channel to voltage difference channel 1 vs 2');
end

if isequal(ecgChannel, 'v1')
    ecgChannel = 'v12';
end

if strcmpi(ecgChannel, 'PULS'); % for pulse oxy, take 1st channel as well)
    ecgChannel = 'c1';
end

isSingleChannel = ecgChannel(1) == 'c';
isVoltageDifference = ecgChannel(1) == 'v';

if isSingleChannel
    iChannel = str2num(ecgChannel(2));
    c                   = cPerChannelInterpolated{iChannel};
    
    % TODO: strictly speaking, these have to be interpolated as well
    cSignals            = cSignalsPerChannel{iChannel};
    extTriggerSignals   = extTriggerSignalsPerChannel{iChannel};
    
elseif isVoltageDifference
    iChannel1 = str2num(ecgChannel(2));
    iChannel2 = str2num(ecgChannel(3));
    c                   = cPerChannelInterpolated{iChannel1}-cPerChannelInterpolated{iChannel2};
    
    % TODO: strictly speaking, these have to be interpolated as well
    cSignals            = cSignalsPerChannel{iChannel1};
    extTriggerSignals   = extTriggerSignalsPerChannel{iChannel1};
    
else
    
    switch lower(ecgChannel)
        case 'mean' % of v1 and v2
            c                   = (cPerChannelInterpolated{1}-cPerChannelInterpolated{2} + ...
                cPerChannelInterpolated{3}-cPerChannelInterpolated{4})/2;
            cSignals            = cSignalsPerChannel{1};
            extTriggerSignals   = extTriggerSignalsPerChannel{1};
    end
end


%% Show some plots of (interpolated) channel data
if DEBUG
    fh = tapas_physio_get_default_fig_params();
    set(fh, 'Name', 'Read-In: Raw ECG Siemens Tics data, split into channels')
    verbose.fig_handles(end+1) = fh;
    
    
    %% first subplot with raw channels
    hAx(1) = subplot(2,1,1);
    
    hp = [];
    strLegend = {};
    for iChannel = 1:nChannels
        hp(end+1) = plot((cTics-cTics(1))*dtTics,cPerChannelInterpolated{iChannel}); hold all;
        strLegend{end+1} = sprintf('iChannel %d', iChannel);
        
        if ~isempty(cSignalsPerChannel{iChannel})
            idxPlotsSignals = find(cSignalsPerChannel{iChannel});
            hp(end+1) = stem((cTicsPerChannel{iChannel}(idxPlotsSignals)-cTics(1))*dtTics...
                , max(c)*ones(size(idxPlotsSignals)));
            strLegend{end+1} = sprintf('Cardiac Pulses iChannel %d', iChannel);
        end
        
        if ~isempty(extTriggerSignalsPerChannel{iChannel})
            idxPlotsSignals = find(extTriggerSignalsPerChannel{iChannel});
            hp(end+1) =stem((cTicsPerChannel{iChannel}(idxPlotsSignals)-cTics(1))*dtTics, max(c)*ones(size(idxPlotsSignals)));
            strLegend{end+1} = sprintf('External Pulses iChannel %d', iChannel);
        end
        
    end
    legend(hp, strLegend{:});
    title(sprintf('Raw Cardiac Channel data, first tic: %d', cTics(1)));
    
    %% second subplot with combined channel
    hAx(2) = subplot(2,1,2);
    
    hp = [];
    strLegend = {};
    
    % also plot selected Channel
    hp(end+1) = plot((cTics-cTics(1))*dtTics, c); hold all;
    strLegend{end+1} = sprintf('selected channel: %s', ecgChannel);
    
    if ~isempty(cSignals)
        idxPlotsSignals = find(cSignals);
        hp(end+1) = stem((cTics(idxPlotsSignals)-cTics(1))*dtTics, ...
            max(c)*ones(size(idxPlotsSignals)));
        strLegend{end+1} = sprintf('Cardiac Pulses selected channel');
    end
    
    if ~isempty(extTriggerSignals)
        idxPlotsSignals = find(extTriggerSignals);
        hp(end+1) =stem((cTics(idxPlotsSignals)-cTics(1))*dtTics, ...
            max(c)*ones(size(idxPlotsSignals)));
        strLegend{end+1} = sprintf('External Pulses selected channel');
    end
    
    legend(hp, strLegend{:});
    
    linkaxes(hAx, 'x');
    title(sprintf('Selected Channel (%s) data', ecgChannel));
    
end