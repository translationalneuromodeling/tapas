function [rpulset, verbose] = tapas_physio_filter_respiratory(...
    rpulset, rsampint, cutoff_freqs, despike, normalize, verbose)
% Preprocesses respiratory data
%
% Key steps
%   + Remove NaNs and outliers
%   + Optional: Despike with sliding-window median filter
%   + Detrend at `cutoff_freqs(1)` Hz
%   + Remove noise above `cutoff_freqs(2)` Hz
%   + Optional: Normalise amplitude
%
% EXAMPLES
%   rpulset = tapas_physio_filter_respiratory(rpulset, rsampint)
%   rpulset = tapas_physio_filter_respiratory( ...
%       rpulset, rsampint, cutoff_freqs, despike, normalize)
%   [rpulset, verbose] = tapas_physio_filter_respiratory( ...
%       rpulset, rsampint, [], [], [], verbose)
%
% INPUTS
%   rpulset         Respiratory timeseries
%   rsampint        Time between successive samples
% OPTIONAL INPUTS
%   cutoff_freqs    [high-pass, low-pass] cutoff frequencies
%                   Default: [0.01, 2.0]
%   despike         Optionally, data is despiked with a median filter
%                   Default: false
%   normalize       Optionally, data is normalized to be in -1...+1 range
%                   Default: true
%   verbose         See `physio.verbose`
%
% OUTPUTS
%   rpulset         Filtered respiratory timeseries
%   verbose         See `physio.verbose`

% Author: Sam Harrison, 2020
% Copyright (C) 2020 TNU, Institute for Biomedical Engineering, University of Zurich and ETH Zurich.
%
% This file is part of the PhysIO toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

if isempty(rpulset)
    rpulset = [];
    return;
end

if (nargin < 3) || isempty(cutoff_freqs)
    cutoff_freqs = [0.01, 2.0];
end
if nargin < 4 || isempty(despike)
    despike = false;
end
if nargin < 5 || isempty(normalize)
    normalize = true;
end
if nargin < 6
    verbose.level = 0;
    verbose.fig_handles = [];
end

%% Basic preproc and outlier removal

% If rpulset has nans, replace them with mean of valid values
try
    rpulsetOffset = mean(rpulset, 'omitnan');
catch % for backwards compatibility < Matlab 2016a
    rpulsetOffset = nanmean(rpulset);
end
rpulset(isnan(rpulset)) = rpulsetOffset;

rpulset = detrend(rpulset, 3);  % Demean / detrend to reduce edge effects

 m = mean(rpulset); s = std(rpulset);
 t = linspace(0.0, rsampint * (length(rpulset) - 1), length(rpulset));


% Now do a check for any gross outliers relative to the statistics of the
% whole timeseries
z_thresh = 5.0;  % Relatively high, as distribution is typically skewed
% figure(); histogram(rpulset);
mpulse = mean(rpulset);
stdpulse = 1.4826 * mad(rpulset, 1);  % Robust to outliers: https://en.wikipedia.org/wiki/Robust_measures_of_scale
outliers = (rpulset > (mpulse + (z_thresh * stdpulse)));
rpulset_out = rpulset;
rpulset_out(outliers) = mpulse + (z_thresh * stdpulse);
outliers = (rpulset_out < (mpulse - (z_thresh * stdpulse)));
rpulset_out(outliers) = mpulse - (z_thresh * stdpulse);
% if verbose.level>=3
%     plot([t(1), t(end)], [z_thresh, z_thresh], 'Color', [0.7, 0.7, 0.7]);
%     plot([t(1), t(end)], [-z_thresh, -z_thresh], 'Color', [0.7, 0.7, 0.7]);
% end

% And despike via a sliding-window median filter
if despike
    mad_thresh = 5.0;  % Again, relatively high so only get large spikes (low-pass filter gets the rest)
    n_pad = ceil(0.25 / rsampint);  % 0.5 s total window length
    rpulset_padded = padarray(rpulset_out, n_pad, 'symmetric');
    medians = movmedian(rpulset_padded, 2 * n_pad + 1, 'Endpoints', 'discard');
    mads = movmad(rpulset_padded, 2 * n_pad + 1, 'Endpoints', 'discard');
    outliers = (abs(rpulset_out - medians) > mad_thresh * mads);
    rpulset_out(outliers) = medians(outliers);
    % if verbose.level>=3
    %     plot(t, (medians - m) / s, 'Color', [0.7, 0.7, 0.7]);
    %     plot(t, (medians + mad_thresh * mads - m) / s, 'Color', [0.7, 0.7, 0.7]);
    %     plot(t, (medians - mad_thresh * mads - m) / s, 'Color', [0.7, 0.7, 0.7]);
    % end
end


%% Detrend and remove noise via filtering

% Filter properties
sampfreq = 1 / rsampint; % Hz
n_pad = ceil(4.0 * (1.0 / cutoff_freqs(1)) * sampfreq); % Generous padding either side

% Low-pass filter to estimate trend
% Then subtract to imitate high-pass filter
% This is typically much more stable than a bandpass filter
d = designfilt( ...
    'lowpassiir', 'HalfPowerFrequency', cutoff_freqs(1), ...
    'FilterOrder', 20, 'SampleRate', sampfreq);

% Use a large padding, and window so tapers back to mean naturally
padding_window = window(@blackmanharris, 2 * n_pad + 1);
rpulset_padded = padarray(rpulset_out, n_pad, 'symmetric');
rpulset_padded(1:n_pad) = padding_window(1:n_pad) .* rpulset_padded(1:n_pad);
rpulset_padded(end-n_pad+1:end) = padding_window(end-n_pad+1:end) .* rpulset_padded(end-n_pad+1:end);

trend = filtfilt(d, rpulset_padded);

% DEBUG
% figure('Name', 'rpulset_padded'); plot(rpulset_padded); hold on; plot(trend)

trend = trend(n_pad+1:end-n_pad);
rpulset_out_trend = rpulset_out - trend;


% Low-pass filter to remove noise
d = designfilt( ...
    'lowpassiir', 'HalfPowerFrequency', cutoff_freqs(2), ...
    'FilterOrder', 20, 'SampleRate', sampfreq);
rpulset_out_trend_filt = filtfilt(d, padarray(rpulset_out_trend, n_pad, 'symmetric'));
rpulset_out_trend_filt = rpulset_out_trend_filt(n_pad+1:end-n_pad);


%% Normalise, if requested

if normalize
    rpulset_out_trend_filt = rpulset_out_trend_filt/max(abs(rpulset_out_trend_filt));
end

%%

 % save relevant variavles for retrospective plotting
 verbose.review.resp_filter.rpulset = rpulset;
 verbose.review.resp_filter.rsampint = rsampint;
 verbose.review.resp_filter.m = m;
 verbose.review.resp_filter.s = s;
 verbose.review.resp_filter.t = t;
 verbose.review.resp_filter.rpulset_out = rpulset_out;
 verbose.review.resp_filter.rpulset_out_trend = rpulset_out_trend;
 verbose.review.resp_filter.trend = trend;
 verbose.review.resp_filter.rpulset_out_trend_filt = rpulset_out_trend_filt;

% Debug and plot
if verbose.level>=3
    [verbose] = tapas_physio_plot_filter_respiratory(rpulset,m, s, t, ...
        rpulset_out, rpulset_out_trend,trend,rpulset_out_trend_filt, verbose);

end

rpulset = rpulset_out_trend_filt;
end