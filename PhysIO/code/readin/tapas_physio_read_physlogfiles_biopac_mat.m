function [c, r, t, cpulse, acq_codes] = tapas_physio_read_physlogfiles_biopac_mat(log_files, ...
    cardiac_modality, verbose)
% reads out physiological time series (ECG, PMU, resp belt) and timing vector for Biopac .mat
% files
%
%   [cpulse, rpulse, t, c] = tapas_physio_read_physlogfiles_GE(logfiles, ...
%                               verbose)
%
%   NOTE: if one
%
% IN    log_files
%       .log_cardiac        contains ECG or pulse oximeter time course
%                           for Biopac: usually the same as respiration
%       .log_respiration    contains breathing belt amplitude time course
%                           for Biopac: usually the same as cardiac
%       .sampling_interval  is ignored here, read from logfile (variable
%                           isi) instead
%
%   cardiac_modality
%                       'ecg1_filtered'     filtered 1st ECG channel signal
%                                           (Default)
%                       'ecg2_filtered'     filteered 2nd ECG channel
%                                           (sometimes less gradient artifacts)
%                       'ecg1_raw'          raw 1st ECG channel
%                       'OXY'/'PPU'         pulse plethysmographic unit
%                                           (PPU) signal
%       verbose
%       .level              debugging plots are created if level >=3
%       .fig_handles        appended by handle to output figure
%
% OUT
%   r                   respiratory time series
%   c                   cardiac time series (ECG or pulse oximetry)
%   t                   vector of time points (in seconds)
%   cpulse              time events of R-wave peak in cardiac time series (seconds)
%                       for Biopac: usually empty, kept for compatibility
%   acq_codes           slice/volume start events marked by number <> 0
%                       for time points in t
% EXAMPLE
%   [ons_secs.cpulse, ons_secs.rpulse, ons_secs.t, ons_secs.c] =
%       tapas_physio_read_physlogfiles_GE(logfiles);
%
%   See also tapas_physio_main_create_regressors

% Author: Lars Kasper
% Created: 2016-08-15
% Copyright (C) 2016 Institute for Biomedical Engineering, ETH/Uni Zurich.
%
% This file is part of the PhysIO toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.


%% read out parameters
DEBUG = verbose.level >= 3;

hasRespirationFile = ~isempty(log_files.respiration);
hasCardiacFile = ~isempty(log_files.cardiac);

if hasCardiacFile
    logfile = log_files.cardiac;
elseif hasRespirationFile
    logfile = log_files.respiration;
end

try
    load(logfile, 'data', 'isi', 'isi_units', 'labels', 'start_sample', 'units');
catch
    error(['%s is not a valid Biopac .mat file. Please check it ', ...
        'contains variables: data, isi, isi_units, labels, start_sample and units.'], ...
        logfile);
end

%% interpolate to greater precision, if 2 different sampling rates are given

switch isi_units
    case 'ms'
        dt = isi/1000;
    case 's'
        dt = isi;
end

% column labels:
% 1 ECG100C
% 2 RSP100C
% 3 Digital input
% 4 Digital input   => could be trigger!
% 5 Digital input
% 6 C0 - Comb Band Stop Filter
% 7 C1 - Filter

%% Get respiratory data
labels = cellstr(labels);
iResp = tapas_physio_find_string(labels, 'RSP');
nSamples = size(data, 1);

if ~isempty(iResp)
    r = data(:,iResp);
    
    if ~any(r)
        r = [];
    end
    
else
    r = [];
end


%% Loop over possible cardiac modalities

cardiacModalityArray = ...
    {'ecg1_filtered', 'ecg2_filtered', 'ecg1_raw', 'ecg2_raw', 'ppu'};

% TODO: check for PPU what really is used as a label
labelsCardiacArray = ...
    {'C0 - Comb Band Stop Filter', 'C1 - Filter', 'ECG100C', 'ECG100C', ...
    'ECG100C'};

iChosenModality =  tapas_physio_find_string(cardiacModalityArray, ...
    cardiac_modality);

nModalities = numel(cardiacModalityArray);
indModalities = [iChosenModality, setdiff(1:nModalities, iChosenModality)];

c = [];

iModality = 0;
while isempty(c) && iModality < nModalities
    
    iModality = iModality + 1;
    iCardiac = tapas_physio_find_string(labels, ...
        labelsCardiacArray{indModalities(iModality)});
    
    if ~isempty(iCardiac)
        c = data(:,iCardiac);
        
        if ~any(c)
            c = [];
        end
    end
end


t = -log_files.relative_start_acquisition + ((start_sample + 0:(nSamples-1))*dt)';
cpulse = [];


if DEBUG
    verbose.fig_handles(end+1) = tapas_physio_get_default_fig_params();
    stringTitle = 'Read-In: Extracted time courses from Biopac mat file';
    set(gcf, 'Name', stringTitle);
    plot(t, data);
    legend(labels);
    xlabel('t (seconds)');
    title(stringTitle);
end

%% Determine whether any meaningful trigger column 'Digital Input'

% indTriggers = tapas_physio_find_string(labels, 'Digital input');
indTriggers = tapas_physio_find_string(labels, 'MRTtrigger');

acq_codes = [];
iTrigger = 0;
while isempty(acq_codes) && iTrigger < numel(indTriggers)
    iTrigger = iTrigger + 1;
    
    acq_codes = data(:, indTriggers(iTrigger));
    
    if any(acq_codes)
        [peakHeight, peakLocation] = tapas_physio_findpeaks(acq_codes);
        if ~isempty(peakLocation)
            acq_codes = zeros(nSamples,1);
            if min(peakHeight) >= 5
                acq_codes(peakLocation) = 10; % assumption: volume trigger
            else
                acq_codes(peakLocation) = 1;
            end
        else
            acq_codes = [];
        end
    else
        acq_codes = [];
    end
end


end

