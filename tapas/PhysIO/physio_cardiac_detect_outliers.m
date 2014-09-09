function [outliersHigh,outliersLow,fh] = physio_cardiac_detect_outliers(tCardiac,percentile,deviationPercentUp,deviationPercentDown, ah)
% detects outliers (missed or erroneous pulses) in heartrate given a sequence of heartbeat pulses
%
%   output = physio_cardiac_detect_outliers(input)
%
% IN
%   tCardiac
%   percentile
%   deviationPercentUp
%   deviationPercentDown
%   ah                      axes handle (optional)...specifies where plot is provided
%
% OUT
%   outliersHigh
%   outliersLow
%   fh
%
% EXAMPLE
%   physio_cardiac_detect_outliers
%
%   See also physio_correct_cardiac_pulses_manually
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
% $Id: physio_cardiac_detect_outliers.m 185 2013-04-26 10:55:32Z kasperla $

if nargin < 5
    fh = physio_get_default_fig_params();
    set(fh, 'Name','Diagnostics raw phys time series');
else
    fh = get(ah, 'Parent');
    figure(fh);
    set(fh, 'CurrentAxes', ah);
end
dt = diff(tCardiac);

hp(1) = plot(tCardiac(2:end), dt);
xlabel('t (seconds)');
ylabel('lag between heartbeats (seconds)');
title('temporal lag between heartbeats');
legend('Temporal lag between subsequent heartbeats');

if ~isempty(percentile) && ~isempty(deviationPercentUp) && ~isempty(deviationPercentDown)
    
    nBins = length(dt)/10;
    [dtSort,dtInd]=sort(dt);
    percentile=percentile/100;
    upperThresh=(1+deviationPercentUp/100)*dtSort(ceil(percentile*length(dtSort)));
    lowerThresh=(1-deviationPercentDown/100)*dtSort(ceil((1-percentile)*length(dtSort)));
    outliersHigh=dtInd(find(dtSort>upperThresh));
    outliersLow=dtInd(find(dtSort<lowerThresh));
    
    % plot percentile thresholds
    hold all;
    hp(2) = plot( [tCardiac(2); tCardiac(end)], [upperThresh,upperThresh], 'g--', 'LineWidth',2);
    hp(3) = plot( [tCardiac(2); tCardiac(end)], [lowerThresh,lowerThresh], 'b--', 'LineWidth',2);
    legend('Temporal lag between subsequent heartbeats', 'Upper threshold for selecting outliers', ...
        'Lower threshold for selecting outliers');

    
    if ~isempty(outliersHigh)
        stem(tCardiac(outliersHigh+1),upperThresh*ones(size(outliersHigh)),'g');
        text(tCardiac(2),max(dt),...
            {'Warning: There seem to be skipped heartbeats in the sequence of pulses', ...
            sprintf('first at timepoint %01.1f s',tCardiac(min(outliersHigh+1)))}, ...
            'Color', [0 0.5 0]);
    end
    
    if ~isempty(outliersLow)
        stem(tCardiac(outliersLow+1),lowerThresh*ones(size(outliersLow)),'b');
        text(tCardiac(2), min(dt),...
            {'Warning: There seem to be wrongly detected heartbeats in the sequence of pulses', ...
            sprintf('first at timepoint %01.1f s',tCardiac(min(outliersLow+1)))}, ...
            'Color', [0 0 1]);
    end
    
end

end