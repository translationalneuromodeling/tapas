function [cpulse, verbose] = tapas_physio_get_oxy_pulses_filtered(c, t, ...
            dt120, verbose)
% Determines peaks of pulse oximeter data after Gaussian and high pass
% filtering, thresholding and assuming maximum heart rate (minimum peak
% distance)
%
%   [cpulse, verbose] = tapas_physio_get_oxy_pulses_filtered(c, t, ...
%            dt120, verbose);
%
% IN
%   c               [nSamples, 1] raw pulse oximeter samples
%   t               [nSamples, 1] time vector corresponding to samples (in seconds)
%   dt120           number of samples corresponding to a heart rate of 120 beats
%                   per minutes, i.e. number of samples acquiredi 0.5 seconds
%   verbose         Substructure of PhysIO, holding verbose.level and
%                   verbose.fig_handles with plotted figure handles
%                   debugging plots for thresholding are only provided, if verbose.level >=2
%
% OUT
%   cpulse          time points (seconds) of detected cardiac pulses
%   (heartbeat events, e.g. R-peaks)
%   verbose         Substructure of PhysIO, augmentedy by the additional
%                   figure handles created during this function
%
% EXAMPLE
%   tapas_physio_get_oxy_pulses_filtered
%
%   See also
%
% Author: Lars Kasper
% Created: 2014-08-03
% Copyright (C) 2014 TNU, Institute for Biomedical Engineering, University of Zurich and ETH Zurich.
%
% This file is part of the TAPAS PhysIO Toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.
%
% $Id: tapas_physio_get_oxy_pulses_filtered.m 524 2014-08-13 16:21:56Z kasperla $
dt = t(2) - t(1);
c = c-mean(c); c = c./max(c); % normalize time series

% smooth noisy pulse oximetry data to detect peaks
w = gausswin(dt120,1);
sc = conv(c, w, 'same');
sc = sc-mean(sc); sc = sc./max(sc); % normalize time series

% Highpass filter to remove drifts
cutoff = 1/dt; %1 seconds/per sampling units
forder = 2;
[b,a] = butter(forder,2/cutoff, 'high');
sc =filter(b,a, sc);
sc = sc./max(sc);

[tmp, cpulse] = tapas_physio_findpeaks(sc, 'minpeakheight',...
    thresh_cardiac.min, 'minpeakdistance', dt120);

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

