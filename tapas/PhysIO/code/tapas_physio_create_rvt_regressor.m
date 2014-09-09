function [convRVTOut, rvtOut, verbose] = tapas_physio_create_rvt_regressor(...
    ons_secs, sqpar, verbose)
% computes respiratory response function regressor and respiratory volume per time
%
%    [convHRV, hr] = tapas_physio_create_rvt_regressor(ons_secs, sqpar )
%
% Reference:
%   Birn, R.M., Smith, M.A., Jones, T.B., Bandettini, P.A., 2008.
%       The respiration response function: The temporal dynamics of
%       fMRI signal fluctuations related to changes in respiration.
%       NeuroImage 40, 644?654.
%
% IN
%   ons_secs.
%       fr              filtered respiratory signal time series
%       spulse_per_vol  See also tapas_physio_get_sample_points
%   sqpar.
%       onset_slice
% OUT
%   convRVT             respiratory response function regressor after convolution . See
%                       also
% EXAMPLE
%   [convHRV, hr] = tapas_physio_create_hrv_regressor(physio_out.ons_secs, physio_out.sqpar);
%
%   See also tapas_physio_rvt tapas_physio_rrf
%
% Author: Lars Kasper
% Created: 2014-01-20
% Copyright (C) 2014 TNU, Institute for Biomedical Engineering, University of Zurich and ETH Zurich.
%
% This file is part of the physIO toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.
%
% $Id: tapas_physio_create_rvt_regressor.m 531 2014-08-14 16:58:12Z kasperla $
if nargin < 3
    verbose.level = 0;
    verbose.fig_handles = [];
end

slicenum = 1:sqpar.Nslices;

sample_points  = tapas_physio_get_sample_points(ons_secs, sqpar, slicenum);
rvt = tapas_physio_rvt(ons_secs.fr, ons_secs.t, sample_points, verbose);
rvt = rvt/max(rvt); % normalize for reasonable range of regressor

if verbose.level >=2
    verbose.fig_handles(end+1) = tapas_physio_get_default_fig_params();
    set(gcf, 'Name', 'Convolution Respiration RVT X RRF');
    subplot(2,2,1)
    plot(sample_points,rvt, 'g');xlabel('time (seconds)');ylabel('respiratory volume per time (a. u.)');
end

% create convolution for whole time series first...
dt = sqpar.TR/sqpar.Nslices;
t = 0:dt:50; % 50 seconds regressor
rrf = tapas_physio_rrf(t);
rrf = rrf/max(abs(rrf));
% crf = spm_hrf(dt);
if verbose.level >= 2
    subplot(2,2,2)
    plot(t, rrf,'g');xlabel('time (seconds)');ylabel('respiratory response function');
end

% NOTE: the removal of the mean was implemented to avoid over/undershoots
% at the 1st and last scans of the session due to convolution
convRVT = conv(rvt-mean(rvt), rrf, 'same');

if verbose.level >= 2
    subplot(2,2,3)
    plot(sample_points, convRVT,'g');xlabel('time (seconds)');ylabel('resp vol time X resp response function');
end


% resample to slices needed
nSampleSlices = numel(sqpar.onset_slice);
nScans = numel(sample_points(sqpar.onset_slice:sqpar.Nslices:end));

rvtOut = zeros(nScans,nSampleSlices);
convRVTOut = zeros(nScans,nSampleSlices);
samplePointsOut = zeros(nScans,nSampleSlices);
for iSlice = 1:nSampleSlices
    onset_slice = sqpar.onset_slice(iSlice);
    rvtOut(:,iSlice) = rvt(onset_slice:sqpar.Nslices:end)';
    convRVTOut(:,iSlice) = convRVT(onset_slice:sqpar.Nslices:end);
    samplePointsOut(:,iSlice) = sample_points(onset_slice:sqpar.Nslices:end);
end

if verbose.level >= 2
    subplot(2,2,4)
    hp{1} = plot(samplePointsOut, rvtOut,'k--');hold all;
    hp{2} = plot(samplePointsOut, convRVTOut,'g');
    xlabel('time (seconds)');ylabel('regessor');
    legend([hp{1}(1), hp{2}(1)], 'respiratory response regressor', 'respiratory volume time (a. u.)');
end
