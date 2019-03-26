function [cpulse, verbose] = tapas_physio_findpeaks_template_xcorr(...
    c, pulseCleanedTemplate, cpulseSecondGuess, averageHeartRateInSamples, ...
    verbose, varargin)
% Finds peaks of a time series via pre-determined template via maxima of
% matlab cross correlations (xcorr) via going backward from search starting 
% point in time series, and afterwards forward again
%
%   [cpulse, verbose] = tapas_physio_findpeaks_template_correlation(...
%       c, pulseCleanedTemplate, cpulseSecondGuess, ...
%           averageHeartRateInSamples, verbose)
%
% IN
%   varargin    property name/value pairs for additional options
%
%
% OUT
%
% EXAMPLE
%   tapas_physio_findpeaks_template_correlation
%
%   See also

% Author: Steffen Bollmann, cleanup, xcorr: Lars Kasper
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


nSamples = size(c,1);

debug = verbose.level >= 4;

idxStartPeakSearch = [0 20];

halfTemplateWidthInSamples = floor(numel(pulseCleanedTemplate)/2);

[~,zTransformedTemplate] = tapas_physio_corrcoef12(pulseCleanedTemplate,...
    pulseCleanedTemplate);
isZTransformed = [0 1];

% start and end point of search for representative start cycle
centreSampleStart = round(2*halfTemplateWidthInSamples+1);

if idxStartPeakSearch(1) > 0
    centreSampleStart = centreSampleStart + ...
        cpulseSecondGuess(idxStartPeakSearch(1));
end
centreSampleEnd = cpulseSecondGuess(idxStartPeakSearch(2));

iSignalStart = centreSampleStart - halfTemplateWidthInSamples;
iSignalEnd = centreSampleEnd + halfTemplateWidthInSamples;
signalPart = c(iSignalStart:iSignalEnd);

similarityToTemplate = xcorr(zTransformedTemplate, flipud(signalPart));

% not needed, since only zero-filled
% similarityToTemplate = similarityToTemplate(1:centreSampleEnd);


[C_bestMatch, I_bestMatch] = max(similarityToTemplate);



%% now compute backwards to the beginning:
% go average heartbeat by heartbeat back and look (with
% decreasing weighting for higher distance) for highest
% correlation with template heartbeat

n = I_bestMatch;
bestPosition = n; % to capture case where 1st R-peak is best

peakNumber = 1;

similarityToTemplate = zeros(nSamples,1);

searchStepsTotal    = round(0.5*averageHeartRateInSamples);
searchPositionArray = -searchStepsTotal:searchStepsTotal;
nSamplesSignalPart  = 2*searchStepsTotal+1;
locationWeight      = ones(nSamplesSignalPart,1);

while n > 1+searchStepsTotal+halfTemplateWidthInSamples
    
 
    % Nested function, needs c, zTransformedTemplate, n, halfTemplateWidthInSamples,
    % searchStepsTotal, locationWeight
    similarityToTemplate(n+searchPositionArray) = ...
        get_similarity_to_template();
    
    %find biggest correlation-peak from the last search
    indexSearchStart    = n-searchStepsTotal;
    indexSearchEnd      = n+searchStepsTotal;
    
    indexSearchRange    = indexSearchStart:indexSearchEnd;
    searchRangeValues   = similarityToTemplate(indexSearchRange);
    [C_bestMatch,I_bestMatch] = max(searchRangeValues);
    bestPosition = indexSearchRange(I_bestMatch);
    
    cpulse(peakNumber) = bestPosition;
    peakNumber = peakNumber+1;
    
    
    n=bestPosition-averageHeartRateInSamples;
end % END: going backwards to beginning of time course

%% Now go forward through the whole time series
n           = bestPosition; % 1st R-peak
peakNumber  = 1;
clear cpulse;

% Now correlate template with PPU signal at the positions
% where we would expect a peak based on the average heartrate and
% search in the neighborhood for the best peak, but weight the peaks
% deviating from the initial starting point by a gaussian
searchStepsTotal = round(0.5*averageHeartRateInSamples);

% for weighted searching of max correlation
locationWeight = tapas_physio_gausswin(nSamplesSignalPart);

n = max(n, searchStepsTotal + halfTemplateWidthInSamples + 1);

% zero-pad c at end to allow for detection of last peak by
% template-matching up to the last sample of c
c = [c; zeros(searchStepsTotal + halfTemplateWidthInSamples + 1, 1)];

while n < nSamples % -searchStepsTotal - halfTemplateWidthInSamples
    
     similarityToTemplate(n+searchPositionArray) = ...
         get_similarity_to_template();
    
    %find biggest correlation-peak from the last search
    indexSearchStart    = n - searchStepsTotal;
    indexSearchEnd      = n + searchStepsTotal;
    
    indexSearchRange    = indexSearchStart:indexSearchEnd;
    searchRangeValues   = similarityToTemplate(indexSearchRange);
    [C_bestMatch,I_bestMatch] = max(searchRangeValues);
    bestPosition        = indexSearchRange(I_bestMatch);
    
    cpulse(peakNumber) = bestPosition;
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
        n = bestPosition + currentHeartRateInSamples;
    else
        n = bestPosition + averageHeartRateInSamples;
    end
end


%% Nested function, 
% computes point-wise similarity (cross-correlation) of time course snippet 
% to given peak template (z-transformed)
% nested function to improve performance
%
% needs c, zTransformedTemplate, n, halfTemplateWidthInSamples,
% searchStepsTotal, searchPositionArray, locationWeight
    function similarityToTemplateTmp = get_similarity_to_template()
        % (c, zTransformedTemplate, n, halfTemplateWidthInSamples, searchStepsTotal, locationWeight);
        iSignalStart    = n - halfTemplateWidthInSamples - searchStepsTotal;
        iSignalEnd      = n + halfTemplateWidthInSamples + searchStepsTotal;
        
        signalPart = c(iSignalStart:iSignalEnd);
        similarityToTemplateTmp = ...
            xcorr(zTransformedTemplate, flipud(signalPart));
        similarityToTemplateTmp = similarityToTemplateTmp(1:nSamplesSignalPart);
        % crop beginning and end
        %similarityToTemplateTmp(1:halfTemplateWidthInSamples) = [];
        %similarityToTemplateTmp(end-halfTemplateWidthInSamples+1:end) = [];
        
        % reweight correlations with distance from expected heart beat
        amplitudeWeight = abs(c((n+1)-searchPositionArray));
        similarityToTemplateTmp =  locationWeight.*amplitudeWeight .* ...
            similarityToTemplateTmp;
        
    end

end