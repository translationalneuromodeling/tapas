function [pulseCleanedTemplate, cpulseSecondGuess, averageHeartRateInSamples] = ...
    tapas_physio_get_cardiac_pulse_template(t, c, thresh_min, ...
    dt120, verbose)
% determines cardiac template by a 2-pass peak detection and averaging of
% closest matches to mean and refinements
%
% [pulseCleanedTemplate, cpulseSecondGuess, averageHeartRateInSamples] = ...
%     tapas_physio_get_cardiac_pulse_template(t, c, thresh_min, ...
%    dt120)
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
% $Id: tapas_physio_get_cardiac_pulse_template.m 529 2014-08-14 10:55:16Z kasperla $
% Guess peaks in two steps with updated avereage heartrate
% First step

debug = verbose.level >= 3;
dt = t(2) - t(1);

[tmp, cpulseFirstGuess] = tapas_physio_findpeaks( ...
    c,'minpeakheight',thresh_min,'minpeakdistance', dt120);

hasFirstGuessPeaks = ~isempty(cpulseFirstGuess);


% Second step, refined heart rate estimate
if hasFirstGuessPeaks
    
    averageHeartRateInSamples = round(mean(diff(cpulseFirstGuess)));
    [tmp, cpulseSecondGuess] = tapas_physio_findpeaks(c,...
        'minpeakheight',thresh_min,...
        'minpeakdistance', round(0.5*averageHeartRateInSamples));
    
    if debug
        fh = tapas_physio_get_default_fig_params;
        subplot(3,1,1);
        hold off
        hp(1) = stem(t(cpulseFirstGuess),4*ones(length(cpulseFirstGuess),1),'b');
        hold all;
        hp(2) = stem(t(cpulseSecondGuess),4*ones(length(cpulseSecondGuess),1),'r');
        hp(3) = plot(t, c, 'k');
        legend(hp, ...
            'first guess peaks', 'second guess peaks', 'raw time series');
        title('Finding first peak (heartbeat/max inhale), backwards')
    end
else
    if debug
        fh = tapas_physio_get_default_fig_params;
        subplot(3,1,1);
        plot(t, c, 'k'); title('Finding first peak (heartbeat/max inhale), backwards')
    end
    
end


%% Build template based on the guessed peaks:
% cut out all data around the detected (presumed) R-peaks
%   => these are the representative "QRS"-waves

% template should only be length of a fraction of average heartbeat length    shortenTemplateFactor = 0.5;
shortenTemplateFactor = 0.5; % template should only be length of a fraction of average heartbeat length
halfTemplateWidthInSamples = round(shortenTemplateFactor * ...
    (averageHeartRateInSamples / 2));

% z-transform to have all templates (for averaging) have
% same norm & be mean-corrected
doNormalizeTemplate = true;
nSamplesTemplate = halfTemplateWidthInSamples * 2 + 1;
nPulses = numel(cpulseSecondGuess);
template = zeros(nPulses-3,nSamplesTemplate);
for n=2:nPulses-2
    startTemplate = cpulseSecondGuess(n)-halfTemplateWidthInSamples;
    endTemplate = cpulseSecondGuess(n)+halfTemplateWidthInSamples;
    
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
    subplot(3,1,2);
    tTemplate = dt*(0:2*halfTemplateWidthInSamples);
    plot(tTemplate, template');
    hold all;
    hp(1) = plot(tTemplate, pulseTemplate', '.--g', 'LineWidth', 3, 'Marker', ...
        'o');
    xlabel('t (seconds)');
    title('Templates of physiology time courses per heart beat and mean template');
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
indHighQualityTemplates = find(similarityToTemplate > thresholdHighQualityCorrelation);

% if threshold to restrictive, try with new one: 
% best nMinHighQualityTemplates / nPulses of all found templates used for
% averaging
if numel(indHighQualityTemplates) < nMinHighQualityTemplates
    thresholdHighQualityCorrelation = tapas_physio_prctile(similarityToTemplate, ...
        1 - nMinHighQualityTemplates/nPulses);
    indHighQualityTemplates = find(similarityToTemplate > thresholdHighQualityCorrelation);
end
pulseCleanedTemplate = mean(template(indHighQualityTemplates, :));

if debug
    hp(2) = plot(tTemplate, pulseCleanedTemplate, '.-g', 'LineWidth', 4, ...
        'Marker', 'x');
    legend(hp, 'mean of templates', 'mean of most similar, chosen templates');
end