function [verbose, c_outliers_low, c_outliers_high, r_hist] = tapas_physio_plot_raw_physdata_diagnostics(cpulse, yResp, ...
    thresh_cardiac, verbose, t, c)
% plots diagnostics for raw physiological time series as monitoried by the
% MR scanner breathing belt/ECG

% Author: Lars Kasper
%
% Copyright (C) 2013, Institute for Biomedical Engineering, ETH/Uni Zurich.
%
% This file is part of the PhysIO toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.



%% Cardiac analysis of heartbeat rates

hasCardiacData = ~isempty(cpulse);
hasRespData = ~isempty(yResp);

isVerbose = verbose.level > 0;

if isVerbose

    fh = tapas_physio_get_default_fig_params(verbose);

    verbose.fig_handles(end+1) = fh;
    set(fh, 'Name','Preproc: Diagnostics for raw physiological time series');
    ah = subplot(2,1,1);
    
    if hasCardiacData
        % plot raw cardiac time series, normalized, first
        c = c-mean(c);
        c = c/max(abs(c));
        
        nPulses = numel(cpulse);
        timeCpulse = zeros(nPulses,1);
        for iPulse = 1:nPulses % find sample points in t/c of cpulse-onsets
            [tmp, timeCpulse(iPulse)] = min(abs(t-cpulse(iPulse)));
        end
        plot(t, c, 'Color', [1 0.8, 0.8], 'LineWidth', 1) ; hold on;
        stem(cpulse, c(timeCpulse), 'r', 'LineWidth', 1);
        title('Temporal lag between subsequent heartbeats (seconds)');
    end
else 
    ah = [];
end

if hasCardiacData
    percentile = thresh_cardiac.percentile;
    upperThresh = thresh_cardiac.upper_thresh;
    lowerThresh = thresh_cardiac.lower_thresh;
    [c_outliers_high, c_outliers_low, verbose] = tapas_physio_cardiac_detect_outliers(...
        cpulse, percentile, upperThresh, lowerThresh, verbose, ah);

    if ~isempty(c_outliers_high)
        c_outliers_high = sort(cpulse(c_outliers_high+1));
    end
    
    if ~isempty(c_outliers_low)
        c_outliers_low = sort(cpulse(c_outliers_low+1));
    end
    
else
    c_outliers_high = [];
    c_outliers_low = [];
end


%% Histogram of breathing amplitudes

if hasRespData
    nBins = max(50, min(500, min(length(unique(yResp))/10, floor(length(yResp)/1000))));
    [r_hist, bins] = hist(yResp, nBins);
    
    if isVerbose
        subplot(2,1,2);
        bar(bins, r_hist, 1);
        title('Histogram of breathing belt amplitudes');
    end
else
    r_hist = [];
end
end
