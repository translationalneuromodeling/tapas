function [convRVTOut, rvtOut, verbose] = tapas_physio_create_rvt_regressors(...
    ons_secs, sqpar, model_rvt, verbose)
% computes respiratory response function regressor and respiratory volume per time
%
%    [convHRV, hr] = tapas_physio_create_rvt_regressors(ons_secs, sqpar )
%
% Reference:
%   Birn, R.M., Smith, M.A., Jones, T.B., Bandettini, P.A., 2008.
%       The respiration response function: The temporal dynamics of
%       fMRI signal fluctuations related to changes in respiration.
%       NeuroImage 40, 644-654.
%
% IN
%   ons_secs.
%       fr              filtered respiratory signal time series
%       spulse_per_vol  See also tapas_physio_get_sample_points
%   sqpar.
%       onset_slice
% OUT
%   convRVTOut          [nScans, nDelays, nSampleSlices]
%                       respiratory response function regressor after
%                       convolution for specified delays and downsampled
%                       to given slices.
% EXAMPLE
%   [convHRV, hr] = tapas_physio_create_hrv_regressor(physio_out.ons_secs, physio_out.sqpar);
%
%   See also tapas_physio_rvt_hilbert tapas_physio_rvt_peaks tapas_physio_rrf

% Author: Lars Kasper
% Created: 2014-01-20
% Copyright (C) 2014 TNU, Institute for Biomedical Engineering, University of Zurich and ETH Zurich.
%
% This file is part of the physIO toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

if nargin < 3
    physio = tapas_physio_new;
    model_rvt = physio.model.rvt;
end

delays = model_rvt.delays;


if nargin < 4
    verbose.level = [];
    verbose.fig_handles = [];
end

slicenum = 1:sqpar.Nslices;


% Calculate RVT
sample_points  = tapas_physio_get_sample_points(ons_secs, sqpar, slicenum);
switch lower(model_rvt.method)
    case 'peaks'
        [rvt, ~, ~, verbose] = tapas_physio_rvt_peaks(ons_secs.fr, ons_secs.t, sample_points, verbose);
    case 'hilbert'
        [rvt, verbose] = tapas_physio_rvt_hilbert(ons_secs.fr, ons_secs.t, sample_points, verbose);
    otherwise
        error('Unrecognised value for ''rvt.method'' (%s)!', model_rvt.method)
end
rvt = rvt / max(abs(rvt)); % normalize for reasonable range of regressor

if verbose.level >=2
    verbose.fig_handles(end+1) = tapas_physio_get_default_fig_params();
    set(gcf, 'Name', 'Model: Convolution Respiration RVT X RRF');
    subplot(2,2,1)
    plot(sample_points,rvt, 'g');xlabel('time (seconds)');
    title('Respiratory volume per time');
    ylabel('a.u.');
end


% Generate RRF
dt = sqpar.TR / sqpar.Nslices;
t = 0:dt:60;  % seconds
rrf = tapas_physio_rrf(t);
rrf = rrf / max(abs(rrf));

if verbose.level >= 2
    subplot(2,2,2)
    plot(t, rrf,'g'); xlabel('time (seconds)');
    title('Respiratory response function');
end


% Convolve and rescale for display purposes
convRVT = tapas_physio_conv(rvt, rrf, 'causal');
convRVT = convRVT / max(abs(convRVT));

if verbose.level >= 2
    subplot(2,2,3)
    plot(sample_points, convRVT,'g');xlabel('time (seconds)');
    title('Resp vol time X resp response function');
end


% Create shifted regressors convolved time series, which is equivalent to
% delayed response functions according to Wikipedia (convolution)
%
% "Translation invariance[edit]
% The convolution commutes with translations, meaning that
%
% \tau_x ({f}*g) = (\tau_x f)*g = {f}*(\tau_x g)\,
% where \tau_x is the translation of the function f by x defined by
% (\tau_x f)(y) = f(y-x).

% remove mean and linear trend to fulfill periodicity condition for
% shifting
convRVT = detrend(convRVT);

% TODO: what happens at the end/beginning of shifted convolutions?
nDelays = numel(delays);
nShiftSamples = ceil(delays/dt);

% resample to slices needed
nSampleSlices = numel(sqpar.onset_slice);
nScans = numel(sample_points(sqpar.onset_slice:sqpar.Nslices:end));

rvtOut = zeros(nScans,nSampleSlices);
convRVTOut = zeros(nScans,nDelays,nSampleSlices);
samplePointsOut = zeros(nScans,nSampleSlices);

for iDelay = 1:nDelays
    convRVTShifted = circshift(convRVT, nShiftSamples(iDelay));
    for iSlice = 1:nSampleSlices
        onset_slice = sqpar.onset_slice(iSlice);
        rvtOut(:,iSlice) = rvt(onset_slice:sqpar.Nslices:end)';
        convRVTOut(:,iDelay,iSlice) = convRVTShifted(onset_slice:sqpar.Nslices:end);
        samplePointsOut(:,iSlice) = sample_points(onset_slice:sqpar.Nslices:end);
    end
end

if verbose.level >= 2
    subplot(2,2,4)
    [tmp, iShiftMin] = min(abs(delays));
    hp{1} = plot(samplePointsOut, rvtOut,'k--');hold all;
    hp{2} = plot(samplePointsOut, squeeze(convRVTOut(:,iShiftMin,:)),'g');
    xlabel('time (seconds)');
    title('RVT regessor');
    legend([hp{1}(1), hp{2}(1)], 'respiratory volume / time (a. u.)', ...
        'respiratory response regressor');
end

end
