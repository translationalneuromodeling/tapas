function [cpulse, verbose, plotData] = tapas_physio_findpeaks_template_correlation(...
    c, pulseCleanedTemplate, cpulseSecondGuess, averageHeartRateInSamples, ...
    verbose, varargin)
% Finds peaks of a time series via pre-determined template via maxima of
% correlations via going backward from search starting point in time
% series, and afterwards forward again
%
%   [cpulse, verbose, plotData] = tapas_physio_findpeaks_template_correlation(...
%       c, pulseCleanedTemplate, cpulseSecondGuess, averageHeartRateInSamples, verbose)
%
% IN
%   varargin    property name/value pairs for additional options
%
%   'idxStartPeakSearch' [1,2] indices of cpulseSecondGuess
%                       giving start and end point of search range for
%                       detection of representative starting cycle, from
%                       which both cycles before and after will be
%                       determined via the pulseCleanedTemplate
%
% OUT
%   cpulse
%   verbose
%   plotData        structure of plot-relevant data for algorithm depiction
%                   fields: searchedAt, amplitudeWeight, locationWeight,
%                   similarityToTemplate
%
% EXAMPLE
%   tapas_physio_findpeaks_template_correlation
%
%   See also

% Author: Steffen Bollmann, merged in this function: Lars Kasper
% Created: 2014-08-05
% Copyright (C) 2014 TNU, Institute for Biomedical Engineering,
%                         University of Zurich and ETH Zurich.
%
% This file is part of the TAPAS PhysIO Toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.



% Determine starting peak for the search:
%   search for a representative R-peak a range of peaks

% TODO: maybe replace via convolution with template? "matched
% filter theorem"?


nSamples = size(c,1);

debug = verbose.level >= 4;

defaults.idxStartPeakSearch = [0 20];
defaults.t = 1:nSamples;

args = tapas_physio_propval(varargin, defaults);
tapas_physio_strip_fields(args);

halfTemplateWidthInSamples = floor(numel(pulseCleanedTemplate)/2);

[tmp,zTransformedTemplate] = tapas_physio_corrcoef12(pulseCleanedTemplate,...
    pulseCleanedTemplate);
isZTransformed = [0 1];

%% Find best (representative) R-peak within first 20(=idxStartPeakSearch)
% cycles to start backwards search
% start and end point of search for representative start cycle
centreSampleStart = round(2*halfTemplateWidthInSamples+1);

if idxStartPeakSearch(1) > 0
    centreSampleStart = centreSampleStart + ...
        cpulseSecondGuess(idxStartPeakSearch(1));
end
centreSampleEnd = cpulseSecondGuess(idxStartPeakSearch(2));


similarityToTemplate = zeros(1, ceil(centreSampleEnd));
for n=centreSampleStart:centreSampleEnd
    startSignalIndex    = n - halfTemplateWidthInSamples;
    endSignalIndex      = n + halfTemplateWidthInSamples;
    
    signalPart = c(startSignalIndex:endSignalIndex);
    similarityToTemplate(n) = tapas_physio_corrcoef12(signalPart, ...
        zTransformedTemplate, isZTransformed);
end

[C_bestMatch, I_bestMatch] = max(similarityToTemplate);
clear similarityToTemplate



%% now compute backwards to the beginning:
% go average heartbeat by heartbeat back and look (with
% decreasing weighting for higher distance) for highest
% correlation with template heartbeat

plotData.searchedAt = zeros(size(c));
plotData.locationWeight  = zeros(size(c));
plotData.amplitudeWeight  = zeros(size(c));

peakNumber = 1;
similarityToTemplate = zeros(nSamples,1);

searchStepsTotal = round(0.5*averageHeartRateInSamples);

% zero-pad phys time course to allow computing correlations for first heartbeat
nSamplesPadded = halfTemplateWidthInSamples + searchStepsTotal + 1;
cPadded = [zeros(nSamplesPadded,1); c; zeros(nSamplesPadded, 1)];
n = I_bestMatch + nSamplesPadded; % index update for padded c

if debug
    set(0,'defaultFigureWindowStyle','normal')
    figure(997);clf
    plot(cPadded)
    xlim([0 1000]);
end

% Stepping backwards through heartbeat intervals
while n > 1+searchStepsTotal+halfTemplateWidthInSamples
    
    % search samples within current heartbeat interval
    for searchPosition = -searchStepsTotal:1:searchStepsTotal
        
        % compute current index
        startSignalIndex    = n - halfTemplateWidthInSamples+searchPosition;
        endSignalIndex      = n + halfTemplateWidthInSamples+searchPosition;
        
        signalPart          = cPadded(startSignalIndex:endSignalIndex);
        
        if debug
            figure(997); hold all;
            plot(startSignalIndex:endSignalIndex, cPadded(startSignalIndex:endSignalIndex));
            xlim([0 1000]);
        end
        
        correlation = tapas_physio_corrcoef12(signalPart,zTransformedTemplate, ...
            isZTransformed);
        
        % weight correlations far away from template center
        % less; since heartbeat to be expected in window center
        % gaussianWindow = tapas_physio_gausswin(2*searchStepsTotal+1);
        %                     currentWeight = gaussianWindow(searchPosition+searchStepsTotal+1);
        
        currentWeight = abs(cPadded(n+searchPosition+1));
        correlationWeighted =  currentWeight .* correlation;
        similarityToTemplate(n+searchPosition) = correlationWeighted;
        
    end % search within heartbeat  
    
    %find largest correlation-peak from all the different search positions
    indexSearchStart=n-searchStepsTotal;
    indexSearchEnd=n+searchStepsTotal;
    
    indexSearchRange=indexSearchStart:indexSearchEnd;
    searchRangeValues=similarityToTemplate(indexSearchRange);
    [C_bestMatch,I_bestMatch] = max(searchRangeValues);
    bestPosition = indexSearchRange(I_bestMatch);
    if debug
        figure(997);
        stem(bestPosition, 2);
    end
    n=bestPosition-averageHeartRateInSamples;
end % END: going backwards to beginning of time course

%% Now go forward through the whole time series
% 1st R-peak, correct for index change with zero-padding
n           = bestPosition;
peakNumber  = 1;
clear cpulse;

% Now correlate template with PPU signal at the positions
% where we would expect a peak based on the average heartrate and
% search in the neighborhood for the best peak, but weight the peaks
% deviating from the initial starting point by a gaussian
searchStepsTotal = round(0.5*averageHeartRateInSamples);

% for weighted searching of max correlation
gaussianWindow = tapas_physio_gausswin(2*searchStepsTotal+1);

if debug
    set(0,'defaultFigureWindowStyle','normal')
    figure(997);clf
    plot(cPadded)
    xlim([0 1000]);
end

nLimit = numel(cPadded)-halfTemplateWidthInSamples-searchPosition;

while n <= nLimit
    %search around peak
    for searchPosition = -searchStepsTotal:1:searchStepsTotal
        
        % check only, if search indices within bounds of time series
        if (n+searchPosition) >= 1 && (n+searchPosition+1) <= nLimit
            
            startSignalIndex = ...
                max(1, n-halfTemplateWidthInSamples+searchPosition);
            endSignalIndex = n+halfTemplateWidthInSamples+searchPosition;
            
            signalPart = cPadded(startSignalIndex:endSignalIndex);
            
            if debug
                figure(997); hold all;
                plot(startSignalIndex:endSignalIndex, cPadded(startSignalIndex:endSignalIndex));
                xlim([0 1000]);
            end
            
            correlation = tapas_physio_corrcoef12(signalPart, ...
                zTransformedTemplate, isZTransformed);
            
            locationWeight = gaussianWindow(searchPosition+searchStepsTotal+1);
            amplitudeWeight = abs(cPadded(n+searchPosition+1));
            correlationWeighted =  locationWeight .* amplitudeWeight .* correlation;
            similarityToTemplate(n+searchPosition) = correlationWeighted;
            
            % collect plot Data
            plotData.locationWeight(n+searchPosition) = locationWeight;
            plotData.amplitudeWeight(n+searchPosition) = amplitudeWeight;
            
            if searchPosition==0
                plotData.searchedAt(n+searchPosition) = 1;
            end
        end
    end
    
    %find largest correlation-peak from all the different search positions
    indexSearchStart = max(1, n-searchStepsTotal);
    indexSearchEnd = min(n+searchStepsTotal, nLimit - 1); % -1 because of index shift similarityToTemplate and cPadded
    
    indexSearchRange=indexSearchStart:indexSearchEnd;
    searchRangeValues=similarityToTemplate(indexSearchRange);
    [C_bestMatch,I_bestMatch] = max(searchRangeValues);
    bestPosition = indexSearchRange(I_bestMatch);
    
    if debug
        figure(997); hold all;
        stem(t(bestPosition),4,'g');
    end
    
    cpulse(peakNumber) = bestPosition-nSamplesPadded;
    peakNumber = peakNumber+1;
    
    %only take the last 20 cpulses to compute the current HeartRate
    foundCpulses = size(cpulse,2);
    
    if  foundCpulses < 3
        currentHeartRateInSamples=averageHeartRateInSamples;
    end
    
    if (foundCpulses < 21) && (foundCpulses >= 3)
        currentHeartRateInSamples = round(mean(diff(cpulse)));
    end
    
    if foundCpulses >= 21
        currentCpulses = cpulse (foundCpulses-20:foundCpulses);
        currentHeartRateInSamples = round(mean(diff(currentCpulses)));
    end
    
    
    %check currentHeartRate
    checkSmaller    = currentHeartRateInSamples > 0.5*averageHeartRateInSamples;
    checkLarger     = currentHeartRateInSamples < 1.5*averageHeartRateInSamples;
    
    %jumpToNextPeakSearchArea
    if (checkSmaller && checkLarger)
        n=bestPosition+currentHeartRateInSamples;
    else
        n=bestPosition+averageHeartRateInSamples;
    end
end

plotData.similarityToTemplate = similarityToTemplate;
