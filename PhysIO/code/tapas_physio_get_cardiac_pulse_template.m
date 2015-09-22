function [pulseCleanedTemplate, cpulse2ndGuess, averageHeartRateInSamples] = ...
    tapas_physio_get_cardiac_pulse_template(t, c, verbose, ...
    varargin)
% determines cardiac template by a 2-pass peak detection and averaging of
% closest matches to mean and refinements
%
% [pulseCleanedTemplate, cpulse2ndGuess, averageHeartRateInSamples] = ...
%     tapas_physio_get_cardiac_pulse_template(t, c, verbose, ...
%     varargin)
%
% IN
%
% OUT
%
% EXAMPLE
%   tapas_physio_get_cardiac_pulse_template
%
%   See also
%
% Author: Lars Kasper
% Created: 2014-08-05
% Copyright (C) 2014 TNU, Institute for Biomedical Engineering, University of Zurich and ETH Zurich.
%
% This file is part of the TAPAS PhysIO Toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.
%
% $Id: tapas_physio_get_cardiac_pulse_template.m 640 2015-01-11 22:03:32Z kasperla $


% template should only be length of a fraction of average heartbeat length   
defaults.shortenTemplateFactor = 0.5; 
defaults.minCycleSamples = ceil(0.5/2e-3);
defaults.thresh_min = 0.4;


args = tapas_physio_propval(varargin, defaults);
tapas_physio_strip_fields(args);


debug = verbose.level >= 3;
dt = t(2) - t(1);

%% Guess peaks in two steps with updated avereage heartrate
% First step

[tmp, cpulse1stGuess] = tapas_physio_findpeaks( ...
    c,'minpeakheight',thresh_min,'minpeakdistance', minCycleSamples);

hasFirstGuessPeaks = ~isempty(cpulse1stGuess);


%% Second step, refined heart rate estimate

stringTitle = 'Iterative Template Creation Single Cycle';
     
if hasFirstGuessPeaks
    
    averageHeartRateInSamples = round(mean(diff(cpulse1stGuess)));
    [tmp, cpulse2ndGuess] = tapas_physio_findpeaks(c,...
        'minpeakheight', thresh_min,...
        'minpeakdistance', round(shortenTemplateFactor*...
        averageHeartRateInSamples));
    
    if debug
        
        nPulses1 = length(cpulse1stGuess);
        nPulses2 = length(cpulse2ndGuess);
        fh = tapas_physio_get_default_fig_params();
        set(fh, 'Name', stringTitle);
        verbose.fig_handles(end+1) = fh;
        subplot(3,1,1);
        hold off
        hp(3) = plot(t, c, 'k');
        hold all;
        hp(1) = stem(t(cpulse1stGuess), ...
            4*ones(nPulses1,1),'b');
        
        hp(2) = stem(t(cpulse2ndGuess),...
            4*ones(nPulses2,1),'r');
     
        stringLegend = {
            sprintf('1st guess peaks (N =%d)', nPulses1), ...
            sprintf('2nd guess peaks (N =%d)', nPulses2), ...
            'raw time series'
            };
    
        legend(hp, stringLegend);
        title('Finding first peak (cycle start), backwards')
        
        
        
        %% Plot difference between detected events
        subplot(3,1,2);
        
        meanLag1 = mean(diff(t(cpulse1stGuess)));
        meanLag2 = mean(diff(t(cpulse2ndGuess)));
        
        plot(t(cpulse1stGuess(2:end)), diff(t(cpulse1stGuess)));
        hold all
        plot(t(cpulse2ndGuess(2:end)), diff(t(cpulse2ndGuess)));
        title('Temporal lag between events')
        
         stringLegend = {
            sprintf('1st Guess (Mean lag duration %3.1f s)', meanLag1), ...
            sprintf('2nd Guess (Mean lag duration %3.1f s)', meanLag2) ...
            };
        
        legend(stringLegend);
    end
else
    if debug
        fh = tapas_physio_get_default_fig_params;
        subplot(3,1,1);
        plot(t, c, 'k'); title('Finding first peak of cycle, backwards')
    end
    
end



%% Build template based on the guessed peaks:
% cut out all data around the detected (presumed) R-peaks
%   => these are the representative "QRS"-waves

halfTemplateWidthInSamples = round(shortenTemplateFactor * ...
    (averageHeartRateInSamples / 2));

% z-transform to have all templates (for averaging) have
% same norm & be mean-corrected
doNormalizeTemplate = true;
nSamplesTemplate = halfTemplateWidthInSamples * 2 + 1;
nPulses = numel(cpulse2ndGuess);
template = zeros(nPulses-3,nSamplesTemplate);

for n=2:nPulses-2
    startTemplate = cpulse2ndGuess(n)-halfTemplateWidthInSamples;
    endTemplate = cpulse2ndGuess(n)+halfTemplateWidthInSamples;
    
    template(n,:) = c(startTemplate:endTemplate);
    
    if doNormalizeTemplate
        template(n,:) = template(n,:) - mean(template(n,:),2);
        
        % std-normalized...
        %template(n,:) = template(n,:)./std(template(n,:),0,2);
        % max-norm:
        template(n,:) = template(n,:)./max(abs(template(n,:)));
    end
    
end

%delete first zero-elements of the template
template(1,:) = [];

% template as average of the found representative waves
pulseTemplate = mean(template);

if debug
    figure(fh);
    subplot(3,1,3);
    tTemplate = dt*(0:2*halfTemplateWidthInSamples);
    plot(tTemplate, template');
    hold all;
    hp(1) = plot(tTemplate, pulseTemplate', '.--g', 'LineWidth', 3, 'Marker', ...
        'o');
    xlabel('t (seconds)');
    title('Templates of cycle time course and mean template');
end

% delete the peaks deviating from the mean too
% much before building the final template
[~, pulseTemplate] = tapas_physio_corrcoef12(pulseTemplate, pulseTemplate);
isZtransformed = [0 1];

nTemplates = size(template,1);
similarityToTemplate = zeros(nTemplates,1);
for n=1:nTemplates
    similarityToTemplate(n) = tapas_physio_corrcoef12(template(n,:),pulseTemplate, ...
        isZtransformed);
end

%% Final template for peak search from best-matching templates

thresholdHighQualityCorrelation = 0.95;

% minimal number of high quality templates to be achieved, otherwise
% enforced
nMinHighQualityTemplates = ceil(0.1 * nPulses); 
indHighQualityTemplates = find(similarityToTemplate > ...
    thresholdHighQualityCorrelation);

% if threshold to restrictive, try with new one: 
% best nMinHighQualityTemplates / nPulses of all found templates used for
% averaging
if numel(indHighQualityTemplates) < nMinHighQualityTemplates
    thresholdHighQualityCorrelation = tapas_physio_prctile(similarityToTemplate, ...
        1 - nMinHighQualityTemplates/nPulses);
    indHighQualityTemplates = find(similarityToTemplate > ...
        thresholdHighQualityCorrelation);
end
pulseCleanedTemplate = mean(template(indHighQualityTemplates, :));

if debug
    hp(2) = plot(tTemplate, pulseCleanedTemplate, '.-g', 'LineWidth', 4, ...
        'Marker', 'x');
    legend(hp, 'mean of templates', 'mean of most similar, chosen templates');
    suptitle(stringTitle);
end