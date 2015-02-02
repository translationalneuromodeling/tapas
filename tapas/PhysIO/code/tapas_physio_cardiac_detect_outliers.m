function [outliersHigh,outliersLow,fh] = ...
    tapas_physio_cardiac_detect_outliers(tCardiac,percentile,...
    deviationPercentUp,deviationPercentDown, isVerbose, ah)
% detects outliers (missed or erroneous pulses) in heartrate given a sequence of heartbeat pulses
%
%   output = tapas_physio_cardiac_detect_outliers(input)
%
% IN
%   tCardiac                [nPulses,1]     onset time of cardiac pulses
%   percentile
%   deviationPercentUp
%   deviationPercentDown
%   isVerbose               if false, only warnings are output, no figures
%   ah                      axes handle (optional)...specifies where plot is provided
%
% OUT
%   outliersHigh
%   outliersLow
%   fh
%
% EXAMPLE
%   tapas_physio_cardiac_detect_outliers
%
%   See also tapas_physio_correct_cardiac_pulses_manually
%
% Author: Lars Kasper
% Created: 2013-04-25
% Copyright (C) 2013 TNU, Institute for Biomedical Engineering, University of Zurich and ETH Zurich.
%
% This file is part of the TNU CheckPhysRETROICOR toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.
%
% $Id: tapas_physio_cardiac_detect_outliers.m 619 2014-12-17 14:54:42Z kasperla $

dt = diff(tCardiac);

if nargin < 5
    isVerbose = 1;
end

if isVerbose
    if nargin < 6
        fh = tapas_physio_get_default_fig_params();
        set(fh, 'Name','Diagnostics raw phys time series');
    else
        fh = get(ah, 'Parent');
        figure(fh);
        set(fh, 'CurrentAxes', ah);
    end
    
    
    
    hp(1) = plot(tCardiac(2:end), dt);
    xlabel('t (seconds)');
    ylabel('lag between heartbeats (seconds)');
    title('temporal lag between heartbeats');
    legend('Temporal lag between subsequent heartbeats');
    
else
    fh = [];
end

if ~isempty(percentile) && ~isempty(deviationPercentUp) && ~isempty(deviationPercentDown)
    
    nBins = length(dt)/10;
    [dtSort,dtInd]=sort(dt);
    percentile=percentile/100;
    upperThresh=(1+deviationPercentUp/100)*dtSort(ceil(percentile*length(dtSort)));
    lowerThresh=(1-deviationPercentDown/100)*dtSort(ceil((1-percentile)*length(dtSort)));
    outliersHigh=dtInd(find(dtSort>upperThresh));
    outliersLow=dtInd(find(dtSort<lowerThresh));
    
    % plot percentile thresholds
    if isVerbose
        hold all;
        hp(2) = plot( [tCardiac(2); tCardiac(end)], [upperThresh,upperThresh], 'g--', 'LineWidth',2);
        hp(3) = plot( [tCardiac(2); tCardiac(end)], [lowerThresh,lowerThresh], 'b--', 'LineWidth',2);
        legend(hp, 'Temporal lag between subsequent heartbeats', 'Upper threshold for selecting outliers', ...
            'Lower threshold for selecting outliers');
    end
    
    if isVerbose && ~isempty(outliersHigh)
        stem(tCardiac(outliersHigh+1),upperThresh*ones(size(outliersHigh)),'g');
        text(tCardiac(2),max(dt),...
            {'Warning: There seem to be skipped heartbeats in the sequence of pulses', ...
            sprintf('first at timepoint %01.1f s',tCardiac(min(outliersHigh+1)))}, ...
            'Color', [0 0.5 0]);
    end
    
    if isVerbose && ~isempty(outliersLow)
        stem(tCardiac(outliersLow+1),lowerThresh*ones(size(outliersLow)),'b');
        text(tCardiac(2), min(dt),...
            {'Warning: There seem to be wrongly detected heartbeats in the sequence of pulses', ...
            sprintf('first at timepoint %01.1f s',tCardiac(min(outliersLow+1)))}, ...
            'Color', [0 0 1]);
    end
    
    nHeartBeatDurations = numel(dt);
    percentageOutliers = (numel(outliersLow) + numel(outliersHigh))/...
        nHeartBeatDurations * 100;
    
    if percentageOutliers > 5
        warning(sprintf(['%4.1f %% of all %d heartbeat durations are below %3.1f s ' ...
            'or above %3.1f s \n - consider refining the pulse detection algorithm!' ...
            'Alternatively, do not model cardiac and interaction noise terms'], ...
            percentageOutliers, nHeartBeatDurations, ...
            lowerThresh, upperThresh));
    end
    
end

end
