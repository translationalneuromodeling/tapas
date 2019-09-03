function [cpulse, verbose] = tapas_physio_get_cardiac_pulses(t, c, ...
    cpulse_detect_options, cardiac_modality, verbose)
% extract heartbeat events from ECG or pulse oximetry time course
%
%   cpulse = tapas_physio_get_cardiac_pulses(t, c, cpulse_detect_options, cardiac_modality, verbose);
%
% IN
%   t                  vector of time series of log file (in seconds, corresponds to c)
%   c                  raw time series of ECG or pulse oximeter
%   cpulse_detect_options      
%                      is a structure with the following elements
%           .method -  'auto_matched', 'manual_template', 'load_from_logfile',
%                       'load_template'
%                      Specifies how to determine QRS-wave from noisy input
%                      data
%           .min -     - for modality 'ECG': [percent peak height of sample QRS wave]
%                      if set, ECG heartbeat event is calculated from ECG
%                      timeseries by detecting local maxima of
%                      cross-correlation to a sample QRS-wave;
%                      leave empty, to use Philips' log of heartbeat event
%                      - for modality 'OXY': [peak height of pulse oxymeter] if set, pulse
%                      oxymeter data is used and minimal peak height
%                      set is used to determined maxima
%           .file
%                    - [default: not set] string of file containing a
%                      reference ECG-peak
%                      if set, ECG peak detection via cross-correlation (via
%                      setting .ECG_min) performed with a saved reference ECG peak
%                      This file is saved after picking the QRS-wave
%                      manually (i.e. if .ECG_min is set), so that
%                      results are reproducible
%           .max_heart_rate_bpm
%                   maximum allowed physiological heart rate (in beats
%                   per minute) for subject; default: 90 bpm
%           .krPeak [false] or true; if true, a user input is
%                   required to specify a characteristic R-peak interval in the ECG
%                   or pulse oximetry time series
%   cardiac_modality    'ECG', 'ECG_WiFi' electrocardiogram (with/without
%                   wireless transmisssion for Philips data)
%                       'OXY'/'PPU' pulse oximetry unit
%
%   verbose         Substructure of Physio, holding verbose.level and
%                   verbose.fig_handles with plotted figure handles
%                   debugging plots for thresholding are only provided, if verbose.level >=2
%
% OUT
%   cpulse          vector of onset-times (in seconds) of occuring heart
%                   beats
%
% EXAMPLE
%   ons_samples.cpulse = tapas_physio_get_cardiac_pulses(ons_secs.c,
%   thresh.cardiac, cardiac_modality, cardiac_peak_file);
%
%   See also tapas_physio_main_create_regressors

% Author: Lars Kasper
% Created: 2013-02-16
%
% Copyright (C) 2013, Institute for Biomedical Engineering, ETH/Uni Zurich.
%
% This file is part of the PhysIO toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.


%% detection of cardiac R-peaks

dt = t(2) - t(1);
minPulseDistanceSamples = ...
    floor((1 / (cpulse_detect_options.max_heart_rate_bpm / 60)) / dt);

if isempty(minPulseDistanceSamples)
    minPulseDistanceSamples = round(0.5 / dt); % heart rate < 120 bpm
end

switch lower(cardiac_modality)
    case 'oxy_old'
        [cpulse, verbose] = tapas_physio_get_oxy_pulses_filtered(c, t, ...
            minPulseDistanceSamples, verbose);
    otherwise % {'oxy','ppu', 'oxy_wifi', 'ppu_wifi','ecg', 'ecg_wifi'} etc., including ecg_raw o
        switch cpulse_detect_options.method
            case 'load_from_logfile'
                verbose = tapas_physio_log('How did you end up here? I better do nothing.', ...
                    verbose, 1);
                cpulse = [];
            case {'manual', 'manual_template', 'load', 'load_template'} % load/determine manual template
                [cpulse, verbose] = ...
                    tapas_physio_get_cardiac_pulses_manual_template(...
                    c, t, cpulse_detect_options, verbose);
            case {'auto', 'auto_template', 'auto_matched'}
                [cpulse, verbose] = ...
                    tapas_physio_get_cardiac_pulses_auto_matched( ...
                    c, t, cpulse_detect_options.min, minPulseDistanceSamples, verbose);
        end %  switch cpulse_detect_options.method
end
