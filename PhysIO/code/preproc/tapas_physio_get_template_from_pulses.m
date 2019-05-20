function [pulseCleanedTemplate, pulseTemplate] = tapas_physio_get_template_from_pulses(...
    c, cpulse, halfTemplateWidthInSamples, verbose, varargin)
% Computes mean pulse template given time stamp of detected pulses and raw
% time series; removes outlier pulses with correlation to initial guess of 
% mean template
%
%   [pulseCleanedTemplate, pulseTemplate] = tapas_physio_get_template_from_pulses(...
%                c, cpulse2ndGuess, halfTemplateWidthInSamples, ...
%                doNormalizeTemplate, dt)
% IN
%   c                   cardiac time series (raw physiological recording)
%   cpulse              index vector (wrt c) of detected pulse events (1= event onset/max)
%   halfTemplateWidthInSamples
%                       half the width in samples of template to be
%                       generated, i.e. numel(pulseTemplate) = 2*this+1
%   verbose             verbose substructure of physio, holds all figure
%                       handles created and verbose.level to trigger visualization
%   
%   optional as propertyName/Value pairs:
%
%   thresholdHighQualityCorrelation
%                           default: 0.95
%                           outliers pulses below that correlation will be 
%                           removed for averaging when retrieving final template
%   minFractionIncludedPulses 
%                           default: 0.1
%                           defines minimum fraction of total number of
%                           found pulses to be included in final template
%                           (after removing outliers). If there are not
%                           enough clean pulses, this value is enforced by
%                           lowering quality thresholds.
%   doNormalizeTemplate     default: true; if true, subtracts mean and
%                           scales max to 1 for each pulse, before
%                           averaging
%   dt                      default: 1; defines time vector only used for 
%                           output plot
%   
%   
% OUT
%   pulseCleanedTemplate    pulseTemplate, mean after removing outliers for
%                           correlation quality
%   pulseTemplate           average of all detected pulse shapes
% EXAMPLE
%   tapas_physio_get_template_from_pulses
%
%   See also tapas_physio_get_cardiac_pulse_template tapas_physio_get_cardiac_pulses_auto_matched
 
% Author:   Lars Kasper
% Created:  2019-01-29
% Copyright (C) 2019 TNU, Institute for Biomedical Engineering,
%                    University of Zurich and ETH Zurich.
%
% This file is part of the TAPAS PhysIO Toolbox, which is released under
% the terms of the GNU General Public License (GPL), version 3. You can
% redistribute it and/or modify it under the terms of the GPL (either
% version 3 or, at your option, any later version). For further details,
% see the file COPYING or <http://www.gnu.org/licenses/>.
 
% z-transform to have all templates (for averaging) have
% same norm & be mean-corrected
defaults.doNormalizeTemplate = true;
defaults.minFractionIncludedPulses = 0.1;
defaults.thresholdHighQualityCorrelation = 0.95;
defaults.dt = 1; % for visualization only

args = tapas_physio_propval(varargin, defaults);
tapas_physio_strip_fields(args);

doDebug = verbose.level >= 3;

nSamplesTemplate = halfTemplateWidthInSamples * 2 + 1;
nPulses = numel(cpulse);
template = zeros(nPulses-3,nSamplesTemplate);

for n=2:nPulses-2
    startTemplate = cpulse(n)-halfTemplateWidthInSamples;
    endTemplate = cpulse(n)+halfTemplateWidthInSamples;
    
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

if doDebug
    fh = verbose.fig_handles(end);
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

%% Final template for peak search from best-matching templates
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


% minimal number of high quality templates to be achieved, otherwise
% enforced
nMinHighQualityTemplates = ceil(minFractionIncludedPulses * nPulses); 
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

if doDebug
    stringTitle = 'Preproc: Iterative Template Creation Single Cycle';
    hp(2) = plot(tTemplate, pulseCleanedTemplate, '.-g', 'LineWidth', 4, ...
        'Marker', 'x');
    legend(hp, 'mean of templates', 'mean of most similar, chosen templates');
    tapas_physio_suptitle(stringTitle);
end