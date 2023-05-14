function [convHRVOut, hrOut, verbose] = tapas_physio_create_hrv_regressors(...
    ons_secs, sqpar, model_hrv, verbose)
% computes cardiac response function regressor and heart rate
%
%    [convHRV, hr] = tapas_physio_create_hrv_regressors(ons_secs, sqpar )
%
% Reference:
%   Chang, Catie, John P. Cunningham, and Gary H. Glover.
%   Influence of Heart Rate on the BOLD Signal: The Cardiac Response Function.
%   NeuroImage 44, no. 3 (February 1, 2009): 857-869.
%   doi:10.1016/j.neuroimage.2008.09.029.
%
% IN
%   ons_secs.
%       cpulse          onset times (seconds) of heartbeat pulses (R-wave peak)
%       spulse_per_vol  See also tapas_physio_get_sample_points
%   sqpar.
%       onset_slice
%
% OUT
%   convHRV             cardiac response function regressor after convolution . See
%                       also
% EXAMPLE
%   [convHRV, hr] = tapas_physio_create_hrv_regressors(physio_out.ons_secs, ...
%                       physio_out.sqpar);
%
%   See also tapas_physio_hr tapas_physio_crf

% Author: Lars Kasper
% Created: 2013-07-26
% Copyright (C) 2013 TNU, Institute for Biomedical Engineering, University of Zurich and ETH Zurich.
%
% This file is part of the physIO toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

if nargin < 3
    physio = tapas_physio_new;
    model_hrv = physio.model.hrv;
end


delays = model_hrv.delays;

if nargin < 4
    verbose.level = [];
    verbose.fig_handles = [];
end

slicenum = 1:sqpar.Nslices;


% Calculate HR
sample_points  = tapas_physio_get_sample_points(ons_secs, sqpar, slicenum);
hr = tapas_physio_hr(ons_secs.cpulse, sample_points);

% Generate CRF
dt = sqpar.TR/sqpar.Nslices;
t = 0:dt:30;  % seconds
crf = tapas_physio_crf(t);
crf = crf / max(abs(crf));

% Convolve and rescale for display purposes
convHRV = tapas_physio_conv(hr, crf, 'causal');
convHRV = convHRV / max(abs(convHRV));

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
convHRV_detrend = detrend(convHRV);

% TODO: what happens at the end/beginning of shifted convolutions?
nDelays = numel(delays);
nShiftSamples = ceil(delays/dt);

% resample to slices needed
nSampleSlices = numel(sqpar.onset_slice);
nScans = numel(sample_points(sqpar.onset_slice:sqpar.Nslices:end));

hrOut = zeros(nScans,nSampleSlices);
convHRVOut = zeros(nScans,nDelays,nSampleSlices);
samplePointsOut = zeros(nScans,nSampleSlices);

for iDelay = 1:nDelays
    convHRVShifted = circshift(convHRV_detrend, nShiftSamples(iDelay));
    for iSlice = 1:nSampleSlices
        onset_slice = sqpar.onset_slice(iSlice);
        hrOut(:,iSlice) = hr(onset_slice:sqpar.Nslices:end)';
        convHRVOut(:,iDelay,iSlice) = convHRVShifted(onset_slice:sqpar.Nslices:end);
        samplePointsOut(:,iSlice) = sample_points(onset_slice:sqpar.Nslices:end);
    end
end

% save relevant structures
verbose.review.create_hrv_regressors.sample_points = sample_points;
verbose.review.create_hrv_regressors.hrOut = hrOut;
verbose.review.create_hrv_regressors.hr = hr;
verbose.review.create_hrv_regressors.t = t;
verbose.review.create_hrv_regressors.crf = crf;
verbose.review.create_hrv_regressors.convHRV = convHRV;
verbose.review.create_hrv_regressors.delays = delays;
verbose.review.create_hrv_regressors.samplePointsOut = samplePointsOut;
verbose.review.create_hrv_regressors.convHRVOut = convHRVOut;

if verbose.level>=2
   [verbose] = tapas_physio_plot_create_hrv_regressors(sample_points, hrOut, ...
    hr, t, crf, convHRV, delays,samplePointsOut, convHRVOut, verbose)

end

end