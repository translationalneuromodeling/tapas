function [cpulse, verbose] = tapas_physio_get_cardiac_pulses(t, c, ...
    thresh_cardiac, cardiac_modality, dt120, verbose)
% extract heartbeat events from ECG or pulse oximetry time course
%
%   cpulse = tapas_physio_get_cardiac_pulses(t, c, thresh_cardiac, cardiac_modality, verbose);
%
% IN
%   t                  vector of time series of log file (in seconds, corresponds to c)
%   c                  raw time series of ECG or pulse oximeter
%   thresh_cardiac      is a structure with the following elements
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
%           .krPeak [false] or true; if true, a user input is
%           required to specify a characteristic R-peak interval in the ECG
%           or pulse oximetry time series
%   cardiac_modality    'ECG', 'ECG_WiFi' electrocardiogram (with/without
%                   wireless transmisssion for Philips data)
%                       'OXY'/'PPU' pulse oximetry unit
%   dt120           - minimum distance between heart beats; default 120
%                   bpm, i.e. 0.5 s
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
%
% Author: Lars Kasper
% Created: 2013-02-16
%
% Copyright (C) 2013, Institute for Biomedical Engineering, ETH/Uni Zurich.
%
% This file is part of the PhysIO toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.
%
% $Id: tapas_physio_get_cardiac_pulses.m 645 2015-01-15 20:41:00Z kasperla $

%% detection of cardiac R-peaks

dt = t(2)-t(1);
if nargin < 5 || isempty(dt120)
    dt120 = round(0.5/dt); % heart rate < 120 bpm
end
switch lower(cardiac_modality)
    case 'oxy_old'
        [cpulse, verbose] = tapas_physio_get_oxy_pulses_filtered(c, t, ...
            dt120, verbose);
    case {'oxy','ppu', 'oxy_wifi', 'ppu_wifi','ecg', 'ecg_wifi'}
        switch thresh_cardiac.method
            case 'load_from_logfile'
                warning('How did you end up here? I better do nothing.');
                cpulse = [];
            case {'manual', 'manual_template', 'load', 'load_template'} % load/determine manual template
                [cpulse, verbose] = ...
                    tapas_physio_get_cardiac_pulses_manual_template(...
                    c, t, thresh_cardiac, verbose);
            case {'auto', 'auto_template', 'auto_matched'}
                [cpulse, verbose] = ...
                    tapas_physio_get_cardiac_pulses_auto_matched( ...
                    c, t, thresh_cardiac.min, dt120, verbose);
        end %  switch thresh_cardiac.method
    otherwise
        disp('How did you measure your cardiac cycle, dude? (ECG, OXY)');
end
