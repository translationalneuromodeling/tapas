function [cpulse, verbose] = tapas_physio_get_cardiac_pulses_auto_matched(...
    c, t, thresh_min, minPulseDistanceSamples, verbose, ...
    methodPeakDetection)
% Automated, iterative pulse detection from cardiac (ECG/OXY) data
% (1) Creates a template of representative heartbeats automatically (as
%     *_auto-function)
% (2) Uses template for peak-detection via matched-filtering of time course
%     with determined filter (as *_manual_template-function)
%
%   [cpulse, verbose] = tapas_physio_get_cardiac_pulses_auto(...
%    c, t, thresh_min, minPulseDistanceSamples, verbose)
%
% IN
%   methodPeakDetection 'correlation' or 'matched'
%                       default: 'correlation' (by Steffen Bollmann),
%                                maximises cross-correlation between
%                                template pulse wave and time course to
%                                detect peak
%                       'matched' experimental, uses template as matched
%                                filter to determine maxima
%
% OUT
%   cpulse              [nPulses,1] vector of time points (in seconds) that
%                       peak was detected (e.g. QRS-wave R-peaks of ECG)
%
% EXAMPLE
%   tapas_physio_get_cardiac_pulses_auto_matched
%
%   See also
%
% Author: Steffen Bollmann, Kinderspital Zurich
% Created: 2014-03-20
% Copyright (C) 2014 TNU, Institute for Biomedical Engineering, University of Zurich and ETH Zurich.
%
% This file is part of the physIO toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.
%
% $Id: tapas_physio_get_cardiac_pulses_auto_matched.m 636 2015-01-10 23:41:56Z kasperla $
if nargin < 5
    verbose.level = 0;
    verbose.fig_handles = [];
end

if nargin < 6
    methodPeakDetection = 'correlation'; %'matched_filter' or 'correlation'
end

c = c-mean(c); c = c./std(c); % normalize time series

%% Determine template for QRS-wave (or pulse) 
[pulseCleanedTemplate, cpulseSecondGuess, averageHeartRateInSamples] = ...
    tapas_physio_get_cardiac_pulse_template(t, c, verbose, ...
    'thresh_min', thresh_min, ...
    'minCycleSamples', minPulseDistanceSamples, ...
    'shortenTemplateFactor', 0.5);

%% Perform peak detection with specific method

switch methodPeakDetection
   
    case 'correlation' % Steffen's forward-backward-correlation
        [cpulse, verbose] = tapas_physio_findpeaks_template_correlation(...
            c, pulseCleanedTemplate, cpulseSecondGuess, averageHeartRateInSamples, ...
            verbose);
       
    case 'matched_filter'
        %ECG_min = thresh_min;
        ECG_min = thresh_min * max(pulseCleanedTemplate);
        kRpeak = pulseCleanedTemplate;
        inp_events = [];
        [cpulse, ECG_min_new, kRpeak] = tapas_physio_find_ecg_r_peaks(t,c, ...
            ECG_min, kRpeak, inp_events);
end

cpulse = t(cpulse);

if verbose.level >=2
    verbose.fig_handles(end+1) = tapas_physio_get_default_fig_params();
    titstr = 'Peak Detection from Automatically Generated Template';
    set(gcf, 'Name', titstr);
    plot(t, c, 'k');
    hold all;
    stem(cpulse,4*ones(size(cpulse)), 'r');
    legend('Raw time course',...
        'Detected maxima (cardiac pulses / max inhalations)');
    title(titstr);
end




