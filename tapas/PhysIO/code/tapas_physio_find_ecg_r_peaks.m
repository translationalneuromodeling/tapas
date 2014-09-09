function [events, ECG_min, kRpeak] = tapas_physio_find_ecg_r_peaks(t,y, ECG_min, kRpeak, inp_events)
% finds ECG-R peaks in a timecourse, if MR-scanner detection failed (esp. at high field >= 7 T)
%
% USAGE
%   [events, ECG_min, kRpeak] = tapas_physio_find_ecg_r_peaks(t,y, ECG_min, kRpeak,
%       inp_events)
% 
% Given a timing vector and the corresponding ECG-timecourse, the user
% chooses a specific R-peak patttern with the mouse, which is then re-found
% in the time series to detect all R-peaks
%
% INPUT
%   t           - time vector
%   y           - ECG time course, same length as t
%   ECG_min     - threshold for smoothed time series (middle plot) to be regarded as R-peak
%   inp_events  - scanner suggestions for R-peaks (plotted for comparison)
%   k           - [optional, otherwise interactively chosen]
%                 characteristic R-peak-wave of a heartbeat(convolution kernel)
%
% OUTPUT
%   events  - indices of t where an R-peak was found
%   ECG_min - threshold for smoothed time series (middle plot) to be regarded as R-peak
%   kRpeak  - used characteristic R-peak-wave of a heartbeat(convolution kernel)
%
% USAGE:
%    [events, ECG_min, kRpeak] = tapas_physio_find_ecg_r_peaks(t,y, ECG_min, [kRpeak], [inp_events])
%
% NOTE: The approach uses a matched-filter smoothing of the time series
% with the selected snippet of the typical R-wave shape as the convolution
% kernel
%
%
% -------------------------------------------------------------------------
% Lars Kasper, September 2011
%
% Copyright (C) 2013 Institute for Biomedical Engineering, ETH/Uni Zurich.
%
% This file is part of the PhysIO toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.
%
% $Id: tapas_physio_find_ecg_r_peaks.m 516 2014-07-17 21:54:50Z kasperla $
%
manual_mode = ~exist('kRpeak', 'var') || isempty(kRpeak);

if manual_mode
    %% Plot ECG curve, central part and already detected input events
    fh = tapas_physio_get_default_fig_params();
    set(fh, 'Name', 'Detection of R-wave from measured ECG-timecourse');
    subplot(3,1,1); hold off;
    plot(t(end/2-3000:end/2+3000), y(end/2-3000:end/2+3000)); hold all;
    if exist('inp_events', 'var');
        ax = axis;
        stem(t(inp_events),ax(4)/2*ones(size(inp_events)));
        axis(ax);
    end
    title('Central snippet of ECG-curve');
    xlabel('t(s)');
    
    %% Interactive mode to identify R-wave snippet
    pause(1);
    title('Please click on a starting point (e.g. minimum) of the characteristic part of a QRS-wave', 'Color', [1 0 0]);
    [I1, J1] = ginput(1);
    hold all; plot(I1,J1, 'b*', 'MarkerSize',10);
    title('Please click on end point of THE SAME characteristic QRS-wave snippet');
    [I2, J2] = ginput(1);
    hold all; plot(I2,J2, 'b*', 'MarkerSize',10);
    title('Central snippet of ECG-curve with chosen QRS-wave filter','Color', [0 0 0]);
    I1 = find(t<I1,1,'last');
    I2 = find(t>I2,1, 'first');
    kRpeak = y(floor(I1):ceil(I2));
end


%% smooth ECG curve with R-wave kernel and plot autocorrelation

sy = conv(y./sqrt(sum(kRpeak.^2)),kRpeak/sqrt(sum(kRpeak.^2)),'same');

peaks_found     = false;
thresh_changed  = false;
dt = t(2) - t(1);
nSamplesBpm120 = floor((60/120)/dt);
% lower threshold until peaks are found in autocorrelation function
while ~peaks_found
    if ECG_min < 0
        [tmp, events] = tapas_physio_findpeaks(-sy,'minpeakheight', -ECG_min, 'minpeakdistance',nSamplesBpm120);
    else
        [tmp, events] = tapas_physio_findpeaks(sy,'minpeakheight', ECG_min, 'minpeakdistance',nSamplesBpm120);
    end
    peaks_found = ~isempty(events);
    if ~peaks_found
        thresh_changed = true;
        ECG_min = ECG_min/2;
    end
end

if thresh_changed
    warning('ECG_min threshold was changed to %f', ECG_min);
end



%% plot found R-peak events after thresholding

if manual_mode
    hs(1) = subplot(3,1,2); hold off
    plot(t,sy); hold all;
    stem(t(events), ECG_min*ones(size(events)));
    plot(t, ECG_min*ones(size(t)));
    xlabel('t(s)');
    legend('smoothed ECG-curve', 'detected R-peaks', 'ECG_{min} - correlation threshold');
    title('Correlation to chosen snippet - ECG smoothed with matched filter of 1 QRS-wave');
    
    
    
    %% Plot R-peaks detected by autocorrelation function
    
    hs(2) = subplot(3,1,3); hold off;
    plot(t,y); hold all;
    stem(t(events), ECG_min*max(kRpeak)*ones(size(events)));
    if exist('inp_events', 'var');
        ax = axis;
        stem(t(inp_events),ax(4)/2*ones(size(inp_events)));
        legend('ECG time course', 'detected R-peaks', 'input: scanner-detected R-peaks');
    end
    title('raw ECG time course with detected heartbeat starts');
    xlabel('t(s)');
    linkaxes(hs,'x');
end
end
