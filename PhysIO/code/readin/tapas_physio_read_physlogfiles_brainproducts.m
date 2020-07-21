function [c, r, t, cpulse, acq_codes] = tapas_physio_read_physlogfiles_brainproducts(log_files, ...
    cardiac_modality, verbose)
% reads out physiological time series (ECG, PMU, resp belt) and timing vector for BrainProducts .eeg file
%
%   [cpulse, rpulse, t, c, acq_codes] = tapas_physio_read_physlogfiles_brainproducts(logfiles, ...
%                               verbose)
%
%
% IN    log_files
%       .cardiac        contains ECG or pulse oximeter time course
%                           for BrainProducts: usually the same as respiration
%       .respiration    contains breathing belt amplitude time course
%                           for BrainProducts: usually the same as cardiac
%       .sampling_interval  is ignored here, read from logfile
%
%   cardiac_modality
%                       'ecg1_filtered'     filtered 1st ECG channel signal
%                                           (Default)
%                       'ecg2_filtered'     filteered 2nd ECG channel
%                                           (sometimes less gradient artifacts)
%                       'ecg1_raw'          raw 1st ECG channel
%
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
% Created: 2017-01-27
% Copyright (C) 2017 Institute for Biomedical Engineering, ETH/Uni Zurich.
%
% This file is part of the PhysIO toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.


%% user input

% volume trigger event
event_type = 'Response';
event_value = 'R128';

switch lower(cardiac_modality)
    case {'ecg','ecg1_filtered'}
        % chose one ecg channel (only for visualisation)
        ecg_ch = 1;
    case 'ecg2_filtered'
       % chose one ecg channel (only for visualisation)
        ecg_ch = 2;
end
 
% is ecg data flipped?
ecg_is_flipped = 1;

%% data

% read data and header info using field trip (included in SPM)
if ~exist('ft_read_header', 'file')
   pathSpm = fileparts(which('spm'));
   addpath(genpath(fullfile(pathSpm, 'external', 'fieldtrip')));
end

hdr = ft_read_header(log_files.cardiac);
data = ft_read_data(log_files.cardiac);

fs = hdr.Fs; % sampling frequency in Hz
N = hdr.nSamples; % number of samples
dt = 1/fs; % sampling interval in seconds
t = linspace(0,dt*(N-1), N); % time vector in seconds

fh = [];

% plot first 10 seconds of the raw data
fh(end+1,1) = tapas_physio_get_default_fig_params(); 
plot_end = 10 * fs;
plot(t(1:plot_end), data(:,(1:plot_end)));
xlabel('time in seconds');
ylabel(hdr.chanunit);
legend(hdr.label);

% extract ECG data
if ecg_is_flipped
    s = -data(ecg_ch,:);
else
    s = data(ecg_ch,:);
end

% plot first 10 seconds again for check
fh(end+1,1) = tapas_physio_get_default_fig_params(); 
plot_end = 10 * fs;
plot(t(1:plot_end), s(:,(1:plot_end)));
xlabel('time in seconds');
ylabel(hdr.chanunit{ecg_ch});
legend(hdr.label{ecg_ch});
%% events
% display all events in the data on the command line
cfg = [];
cfg.dataset = log_files.cardiac;
cfg.trialdef.eventtype  = '?';
ft_definetrial(cfg);

% define volume trigger
cfg.trialdef.eventtype = event_type;
cfg.trialdef.eventvalue = event_value;
cfg = ft_definetrial(cfg);
trigger_pos = cfg.trl(:,1);
n_trigger = length(trigger_pos);

% plot raw data and events
fh(end+1,1) = tapas_physio_get_default_fig_params(); 
max_value = max(s);
plot(t, s);
hold all;
stem(t(trigger_pos), max_value * ones(1,n_trigger));

% find segments
% compute the number of samples between each trigger (pos-1)
diff_trigger_pos = diff(trigger_pos);
% find the positons where a change is happening
diff_diff_trigger_pos = find(diff(diff_trigger_pos));
% add first and last event
start_segment = [1; diff_diff_trigger_pos(2:2:end)+1];
end_segment = [diff_diff_trigger_pos(1:2:end)+1; n_trigger];
% number of samples per trials
n_trial_samples = diff_trigger_pos(start_segment);
% position of start segment in samples
sample_start_segment = trigger_pos(start_segment);
% position of end segment in samples (not including the whole trial)
sample_end_segment = trigger_pos(end_segment);
% number of segments
n_segments = length(start_segment);
% number of trials per segment
n_trials = end_segment - start_segment + 1;
% TR per trial
TR_trials = n_trial_samples*dt;

% plot segments
stem(t(sample_start_segment), max_value * ones(1,n_segments), '--', 'LineWidth', 5);
stem(t(sample_end_segment), max_value * ones(1,n_segments), '--', 'LineWidth', 5);
legend('signal', 'all trigger', 'start segment', 'end segment');

%% load only first segment for sanity check
cfg = [];
cfg.dataset = log_files.cardiac;
cfg.trialdef.eventtype = event_type;
cfg.trialdef.eventvalue = event_value;
cfg.trialfun = 'ft_trialfun_segment';
cfg.trialdef.prestim = 0;
cfg.trialdef.poststim = TR_trials(1);
cfg.trialdef.segment_start = start_segment(1);
cfg.trialdef.segment_end = end_segment(1);
cfg = ft_definetrial(cfg);

% plot segments
fh(end+1,1) = tapas_physio_get_default_fig_params(); 
max_value = max(s);
plot(t, s);
hold all;
stem(t(cfg.trl(:,1)), max_value * ones(1,n_trials(1)));
stem(t(cfg.trl(:,2)), max_value * ones(1,n_trials(1)));
legend('signal', 'start trial', 'end trial');

%% save results
[~,name] = fileparts(file_name);
save_name = fullfile(file_path, [name, '_segments.mat']);
save(save_name, 'sample_start_segment', 'sample_end_segment', ...
    'n_segments', 'n_trials', 'TR_trials', 'start_segment', 'end_segment', ...
    'event_type', 'event_value', 'n_trial_samples', 'fs', ...
    'ecg_is_flipped', 'ecg_ch', 'file_path', 'file_name');