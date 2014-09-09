function [rvt, rpulseMax, rpulseMin, verbose] = ...
    tapas_physio_rvt(fr, t, sample_points, verbose)
% computes respiratory volume per time from filtered time series
%
%    [rvt, rpulseMax, rpulseMin] = tapas_physio_rvt(fr, t)
%
%
% The respiratory volume/time is computed for every time point by taking
% the amplitude difference of the closest local maximum and minimum of the
% breathing cycle and dividing by the distance between two subsequent
% breathing maxima
%
% Reference:
%   Birn, R.M., Smith, M.A., Jones, T.B., Bandettini, P.A., 2008. 
%       The respiration response function: The temporal dynamics of 
%       fMRI signal fluctuations related to changes in respiration. 
%       NeuroImage 40, 644?654.
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
%   See also tapas_physio_create_rvt_regressor
%
% Author: Lars Kasper
% Created: 2014-01-20
% Copyright (C) 2013 TNU, Institute for Biomedical Engineering, University of Zurich and ETH Zurich.
%
% This file is part of the physIO toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.
%
% $Id: tapas_physio_rvt.m 531 2014-08-14 16:58:12Z kasperla $


dt = t(2)-t(1);
dtBreath = round(2/dt); %in seconds, minimum distance between two breaths

% compute breathing "pulses" (occurence times "rpulse" of max inhalation
% times)
thresh_cardiac = [];
thresh_cardiac.min = .1; 
thresh_cardiac.method = 'auto_template';

if nargin < 4
    verbose.level = 0;
    verbose.fig_handles = [];
end

rpulseMax = tapas_physio_get_cardiac_pulses(t, fr, ...
thresh_cardiac,'OXY', dtBreath, verbose);
rpulseMin = tapas_physio_get_cardiac_pulses(t, -fr, ...
thresh_cardiac,'OXY', dtBreath, verbose);
nMax = numel(rpulseMax);
nMin = numel(rpulseMin);
maxFr = max(abs(fr));

if verbose.level>=3 
    verbose.fig_handles(end+1) = tapas_physio_get_default_fig_params();
    set(gcf, 'Name', 'Respiratory Volume per Time');
    plot(t,fr, 'g'); hold all;
    stem(rpulseMax, maxFr*ones(nMax,1),'b');
    stem(rpulseMin, -maxFr*ones(nMin,1),'r');
end

nSamples = length(sample_points);
rv = zeros(nSamples,1);
rvt = rv;

for iSample = 1:nSamples
    ts = sample_points(iSample);
    
    [~,iPulseMax] = min(abs(rpulseMax-ts));
    [~,iPulseMin] = min(abs(rpulseMin-ts)); % could be previous or next exhalation...
    tInhale = rpulseMax(iPulseMax);
    tExhale = rpulseMin(iPulseMin);
    
    [~, iInhale] = min(abs(t-tInhale));
    [~, iExhale] = min(abs(t-tExhale));
    rv(iSample) = abs(fr(iInhale)-fr(iExhale));
    % find next inhalation max and compute time till then
    % (or previous, if already at the end)
    if iPulseMax < nMax
        TBreath = abs(tInhale - rpulseMax(iPulseMax+1));
    else
        TBreath = tInhale - rpulseMax(iPulseMax-1);
    end
    
    rvt(iSample) = rv(iSample)/TBreath;
    
   
end

if verbose.level >=3
    plot(sample_points,rv,'k+');
    plot(sample_points,rvt,'kd');
    legend('filtered breathing signal', 'max inhale', 'max exhale', ...
        'Respiratory volume (RV) at sample points', ...
        'RV per time at sample points');
end
