function fh = tapas_physio_plot_raw_physdata_diagnostics(cpulse, yResp, ...
    thresh_cardiac,  isVerbose, t, c)
% plots diagnostics for raw physiological time series as monitoried by the
% MR scanner breathing belt/ECG
%
% Author: Lars Kasper
%
% Copyright (C) 2013, Institute for Biomedical Engineering, ETH/Uni Zurich.
%
% This file is part of the PhysIO toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.
%
% $Id: tapas_physio_plot_raw_physdata_diagnostics.m 524 2014-08-13 16:21:56Z kasperla $

% cardiac analysis of heartbeat rates

hasCardiacData = ~isempty(cpulse);
hasRespData = ~isempty(yResp);

if isVerbose
    fh = tapas_physio_get_default_fig_params();
    set(fh, 'Name','Diagnostics raw phys time series');
    ah = subplot(2,1,1);
    
    if hasCardiacData
        % plot raw cardiac time series, normalized, first
        c = c-mean(c);
        c = c/max(abs(c));
        
        nPulses = numel(cpulse);
        timeCpulse = zeros(nPulses,1);
        for iPulse = 1:nPulses % find sample points in t/c of cpulse-onsets
            [~,timeCpulse(iPulse)] = min(abs(t-cpulse(iPulse)));
        end
        plot(t, c, 'Color', [1 0.8, 0.8], 'LineWidth', 1) ; hold on;
        stem(cpulse, c(timeCpulse), 'r', 'LineWidth', 1);
    end
else 
    fh = [];
    ah = [];
end

if hasCardiacData
    percentile = thresh_cardiac.percentile;
    upperThresh = thresh_cardiac.upper_thresh;
    lowerThresh = thresh_cardiac.lower_thresh;
    [outliersHigh,outliersLow,fh] = tapas_physio_cardiac_detect_outliers(...
        cpulse, percentile, upperThresh, lowerThresh, isVerbose, ah);
end
title( 'temporal lag between subsequent heartbeats (seconds)');

% histogram of breathing amplitudes



if hasRespData
    nBins = min(length(unique(yResp)), floor(length(yResp)/100));
    
    if isVerbose
        subplot(2,1,2);
        hist(yResp, nBins);
    end
end
title('histogram of breathing belt amplitudes');
end
