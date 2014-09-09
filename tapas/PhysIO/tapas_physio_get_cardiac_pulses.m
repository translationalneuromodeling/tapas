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
%           .modality - 'ecg' or 'oxy'/'ppu'; ECG or Pulse oximeter used?
%           .min -     - for modality 'ECG': [percent peak height of sample QRS wave]
%                      if set, ECG heartbeat event is calculated from ECG
%                      timeseries by detecting local maxima of
%                      cross-correlation to a sample QRS-wave;
%                      leave empty, to use Philips' log of heartbeat event
%                      - for modality 'OXY': [peak height of pulse oxymeter] if set, pulse
%                      oxymeter data is used and minimal peak height
%                      set is used to determined maxima
%           .kRpeakfile
%                    - [default: not set] string of file containing a
%                      reference ECG-peak
%                      if set, ECG peak detection via cross-correlation (via
%                      setting .ECG_min) performed with a saved reference ECG peak
%                      This file is saved after picking the QRS-wave
%                      manually (i.e. if .ECG_min is set), so that
%                      results are reproducible
%           .manual_peak_select [false] or true; if true, a user input is
%           required to specify a characteristic R-peak interval in the ECG
%           or pulse oximetry time series
%   dt120           - minimum distance between heart beats; default 120
%                   bpm, i.e. 0.5 s
%   verbose            debugging plot for thresholding, only provided, if verbose.level >=2
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
% $Id: tapas_physio_get_cardiac_pulses.m 484 2014-05-02 16:53:16Z kasperla $

%% detection of cardiac R-peaks

dt = t(2)-t(1);
if nargin < 5 || isempty(dt120)
    dt120 = round(0.5/dt); % heart rate < 120 bpm
end
switch lower(cardiac_modality)
    case 'oxy_old'
        c = c-mean(c); c = c./max(c); % normalize time series
        
        % smooth noisy pulse oximetry data to detect peaks
        w = gausswin(dt120,1);
        sc = conv(c, w, 'same');
        sc = sc-mean(sc); sc = sc./max(sc); % normalize time series
        
        % Highpass filter to remove drifts
        cutoff=1/dt; %1 seconds/per sampling units
        forder=2;
        [b,a]=butter(forder,2/cutoff, 'high');
        sc =filter(b,a, sc);
        sc = sc./max(sc);
        
        [tmp, cpulse] = tapas_physio_findpeaks(sc,'minpeakheight',thresh_cardiac.min,'minpeakdistance', dt120);
        
        if verbose.level >=2 % visualise influence of smoothing on peak detection
            verbose.fig_handles(end+1) = tapas_physio_get_default_fig_params();
            set(gcf, 'Name', 'PPU-OXY: Tresholding Maxima for Heart Beat Detection');
            [tmp, cpulse2] = tapas_physio_findpeaks(c,'minpeakheight',thresh_cardiac.min,'minpeakdistance', dt120);
            plot(t, c, 'k');
            hold all;
            plot(t, sc, 'r', 'LineWidth', 2);
            hold all
            hold all;stem(t(cpulse2),c(cpulse2), 'k--');
            hold all;stem(t(cpulse),sc(cpulse), 'm', 'LineWidth', 2);
            plot(t, repmat(thresh_cardiac.min, length(t),1),'g-');
            legend('Raw PPU time series', 'Smoothed PPU Time Series', ...
                'Detected Heartbeats in Raw Time Series', ...
                'Detected Heartbeats in Smoothed Time Series', ...
                'Threshold (Min) for Heartbeat Detection');
        end
        
        cpulse = t(cpulse);
        
        
        %courtesy of Steffen Bollmann, KiSpi Zurich
    case {'oxy','ppu'}
        [cpulse, verbose] = tapas_physio_get_cardiac_pulses_auto(c, t, ...
            thresh_cardiac.min, dt120, verbose);
    case 'ecg'
        do_manual_peakfind = true;
        switch thresh_cardiac.method
            case 'auto'
                [cpulse, verbose] = tapas_physio_get_cardiac_pulses_auto(c, t, ...
                    thresh_cardiac.min, dt120, verbose);
            otherwise % load, load_from_logfile, manual
                
                
                % manual peak selection, if no file selected and loading is
                % specified
                
                hasKrpeakLogfile = exist(thresh_cardiac.file,'file') || ...
                    exist([thresh_cardiac.file '.mat'],'file');
                
                % if no file exists, also do manual peak-find
                doSelectTemplateManually = strcmpi(...
                    thresh_cardiac.method, 'manual') || ~hasKrpeakLogfile
                
                if doSelectTemplateManually
                    thresh_cardiac.kRpeak = [];
                    hasECGMin = isfield(thresh_cardiac, 'min') && ~isempty(thresh_cardiac.min);
                    if ~hasECGMin
                        thresh_cardiac.min = 0.5;
                    end
                else
                    fprintf('Loading %s\n', thresh_cardiac.file);
                    ECGfile = load(thresh_cardiac.file);
                    thresh_cardiac.min = ECGfile.ECG_min;
                    thresh_cardiac.kRpeak = ECGfile.kRpeak;
                end
                
                inp_events = [];
                ECG_min = thresh_cardiac.min;
                kRpeak = thresh_cardiac.kRpeak;
                if doSelectTemplateManually
                    while ECG_min
                        [cpulse, ECG_min_new, kRpeak] = tapas_physio_find_ecg_r_peaks(t,c, ECG_min, [], inp_events);
                        fprintf('Press 0, then return, if right ECG peaks were found\n');
                        ECG_min = input('otherwise type next numerical choice for ECG_min and continue the selection: ');
                    end
                else
                    [cpulse, ECG_min_new, kRpeak] = tapas_physio_find_ecg_r_peaks(t,c, ECG_min, kRpeak, inp_events);
                end
                ECG_min = ECG_min_new;
                cpulse = t(cpulse);
                
                % save manually found peak parameters to file
                if doSelectTemplateManually
                    save(thresh_cardiac.file, 'ECG_min', 'kRpeak');
                end
        end %  switch thresh_cardiac.method
    otherwise
        disp('How did you measure your cardiac cycle, dude? (ECG, OXY)');
end
