function [rpulset] = tapas_physio_filter_respiratory(...
    rpulset, rsampint, doNormalize)
% Preprocesses respiratory data
%   + Remove NaNs and outliers
%   + Detrend at 0.01 Hz
%   + Remove noise above 2.0 Hz
%
%   rpulset = tapas_physio_filter_respiratory(pulset,rsampint)
%
% IN
%   rpulset
%   rsamping
%   doNormalize     default:false
%                   Optionally, data is normalized to be in -1...+1 range

% Author: Sam Harrison, 2020
%
% This file is part of the PhysIO toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

if isempty(rpulset)
    rpulset = [];
    return;
end

if nargin < 3
    doNormalize = true;
end

%% Basic preproc

% If rpulset has nans, replace them with zeros
rpulsetOffset = nanmean(rpulset);
rpulset(isnan(rpulset)) = nanmean(rpulset);

rpulset = detrend(rpulset, 3);  % Demean / detrend to reduce edge effects
%figure(); hold all;
%plot(rpulset)

% Now do a check for any outliers
z_thresh = 5.0;  % Relatively high, as distribution is typically skewed
% figure(); histogram(rpulset);
mpulse = mean(rpulset);
stdpulse = std(rpulset);
outliers = (rpulset > (mpulse + (z_thresh * stdpulse)));
rpulset(outliers) = mpulse + (z_thresh * stdpulse);
outliers = (rpulset < (mpulse - (z_thresh * stdpulse)));
rpulset(outliers) = mpulse - (z_thresh * stdpulse);
%plot(rpulset)

%% Detrend and remove noise via filtering

% Filter properties
sampfreq = 1 / rsampint; % Hz
n_pad = ceil(100.0 * sampfreq); % 100.0 s either side

% Low-pass filter to estimate trend
% Then subtract to imitate high-pass filter
% This is typically much more stable than a bandpass filter
d = designfilt( ...
    'lowpassiir', 'HalfPowerFrequency', 0.01, ...
    'FilterOrder', 20, 'SampleRate', sampfreq);
trend = filtfilt(d, padarray(rpulset, n_pad, 'circular'));
trend = trend(n_pad+1:end-n_pad);
%plot(trend)
rpulset = rpulset - trend;
%plot(rpulset)

% Low-pass filter to remove noise
d = designfilt( ...
    'lowpassiir', 'HalfPowerFrequency', 2.0, ...
    'FilterOrder', 20, 'SampleRate', sampfreq);
rpulset = filtfilt(d, padarray(rpulset, n_pad, 'circular'));
rpulset = rpulset(n_pad+1:end-n_pad);
%plot(rpulset)

%% Normalise, if requested

if doNormalize
    rpulset = rpulset/max(abs(rpulset));
end

end