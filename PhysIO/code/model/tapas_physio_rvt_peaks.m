function [rvt, timeRpulseMax, timeRpulseMin, verbose] = ...
    tapas_physio_rvt_peaks(fr, t, sample_points, verbose)
% computes respiratory volume per time from filtered time series
%
%    [rvt, rpulseMax, rpulseMin] = tapas_physio_rvt_peaks(fr, t)
%
%
% NEW:
% The respiratory volume/time is computed by interpolating max/min
% breathing amplitudes between detected peaks and divide them by
% interpolated durations (distances) between max. amplitude breathing
% signals (i.e. max. inhalations)
%
% OLD:
% The respiratory volume/time is computed for every time point by taking
% the amplitude difference of the closest local maximum and minimum of the
% breathing cycle and dividing by the distance between two subsequent
% breathing maxima
%
% Reference:
%   Birn, R.M., Smith, M.A., Jones, T.B., Bandettini, P.A., 2008.
%       The respiration response function: The temporal dynamics of
%       fMRI signal fluctuations related to changes in respiration.
%       NeuroImage 40, 644-654.
%
% IN
%   fr     filtered respiratory amplitude time series
%   t      time vector for fr
%   sample_points       vector of time points (seconds) respiratory volume/time should be calculated
% OUT
%   rvt         respiratory volume per time vector
%   rpulseMax   vector of maximum inhalation time points
%   rpulseMin   vector of minimum inhalation time points
%
% EXAMPLE
%   [rvt, rpulse] = tapas_physio_rvt(fr, t)
%
%   See also tapas_physio_create_rvt_regressor, tapas_physio_rvt_hilbert

% Author: Lars Kasper
% Created: 2014-01-20
% Copyright (C) 2013 TNU, Institute for Biomedical Engineering, University of Zurich and ETH Zurich.
%
% This file is part of the physIO toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.


% compute breathing "pulses" (occurence times "rpulse" of max inhalation
% times)
pulse_detect_options = [];
pulse_detect_options.min = .1;
pulse_detect_options.method = 'auto_matched';
pulse_detect_options.max_heart_rate_bpm = 30;% actually the breathing rate breaths/per minute

if nargin < 4
    verbose.level = 0;
    verbose.fig_handles = [];
end

verbose_no = verbose;
verbose_no.level = 0;
timeRpulseMax = tapas_physio_get_cardiac_pulses(t, fr, ...
    pulse_detect_options, 'OXY', verbose);
timeRpulseMin = tapas_physio_get_cardiac_pulses(t, -fr, ...
    pulse_detect_options, 'OXY', verbose);
nMax = numel(timeRpulseMax);
nMin = numel(timeRpulseMin);
maxFr = max(abs(fr));

[~, iTimeRpulseMax] = ismember(timeRpulseMax, t);
[~, iTimeRpulseMin] = ismember(timeRpulseMin, t);

% interpolate minima an maxima...as in Birn et al., 2006
ampRpulseMax = interp1(timeRpulseMax, fr(iTimeRpulseMax), t, 'linear', 'extrap');
ampRpulseMin = interp1(timeRpulseMin, fr(iTimeRpulseMin), t, 'linear', 'extrap');

% Interpolate breath duration, but don't extrapolate
durationBreath = diff(timeRpulseMax);
interpDurationBreath = interp1( ...
    timeRpulseMax(2:end), durationBreath, t, ...
    'linear');
% Nearest-neighbour interpolation for before/after last breath
% Be more careful here as can't let breath duration go negative
if sum(isnan(interpDurationBreath)) > 0
    nan_inds = isnan(interpDurationBreath);
    interpDurationBreath(nan_inds) = interp1( ...
        timeRpulseMax(2:end), durationBreath, t(nan_inds), ...
        'nearest', 'extrap');
end

if verbose.level>=2
    verbose.fig_handles(end+1) = tapas_physio_get_default_fig_params();
    set(gcf, 'Name', 'Model: Respiratory Volume per Time');
    hp(1) = plot(t,fr, 'g'); hold all;
    stem(timeRpulseMax, maxFr*ones(nMax,1),'b');
    stem(timeRpulseMin, -maxFr*ones(nMin,1),'r');
    hp(2) = plot(t, ampRpulseMax, 'b');
    hp(3) = plot(t, ampRpulseMin, 'r');
    hp(4) = plot(t, interpDurationBreath, 'm');
    strLegend = {
        'filtered breathing signal', ...
        'interpolated max inhalation', 'interpolated max exhalation', ...
        'interpolated breath duration (secs)'};
    legend(hp, strLegend)
end

nSamples = length(sample_points);


method = 'linear';

switch method
    % updated, according to Birn et al. 2006
    case 'linear'
        interpRv = (ampRpulseMax - ampRpulseMin);
        interpRvt = interpRv./interpDurationBreath;
        if verbose.level>=2
            hp(end+1) = plot(t, interpRv,'k');
            hp(end+1) = plot(t, interpRvt, 'c');
            strLegend{end+1} = 'interpolated Respiratory Volume (RV)';
            strLegend{end+1} = 'interpolated Respiratory Volume/Time (RVT)';
            legend(hp, strLegend);
        end
        % deprecated, not using interpolated time-courses
        
        % find nearest neighbor in time vector for sample points, to take
        % these interpolated samples
        iTimeSample = zeros(nSamples,1);
        for iSample = 1:nSamples
            [~, iTimeSample(iSample)] = ...
                min(abs((t - sample_points(iSample))));
        end
        
        rv = interpRv(iTimeSample);
        rvt = interpRvt(iTimeSample);
        
        
    case  'const_interp'
        rv = zeros(nSamples,1);
        rvt = rv;
        
        for iSample = 1:nSamples
            ts = sample_points(iSample);
            
            [~,iPulseMax] = min(abs(timeRpulseMax-ts));
            [~,iPulseMin] = min(abs(timeRpulseMin-ts)); % could be previous or next exhalation...
            tInhale = timeRpulseMax(iPulseMax);
            tExhale = timeRpulseMin(iPulseMin);
            
            [~, iInhale] = min(abs(t-tInhale));
            [~, iExhale] = min(abs(t-tExhale));
            rv(iSample) = abs(fr(iInhale)-fr(iExhale));
            % find next inhalation max and compute time till then
            % (or previous, if already at the end)
            if iPulseMax < nMax
                TBreath = abs(tInhale - timeRpulseMax(iPulseMax+1));
            else
                TBreath = tInhale - timeRpulseMax(iPulseMax-1);
            end
            
            rvt(iSample) = rv(iSample)/TBreath;
            
            
        end
end

if verbose.level >=2
    hp(end+1) = plot(sample_points,rv,'k+');
    hp(end+1) = plot(sample_points,rvt,'cd');
    
    strLegend{end+1} = 'Respiratory volume (RV) at sample points';
    strLegend{end+1} =  'RV per time at sample points';
    legend(hp, strLegend);
end
