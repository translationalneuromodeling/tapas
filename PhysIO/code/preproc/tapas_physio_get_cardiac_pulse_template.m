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

% Author: Lars Kasper
% Created: 2014-08-05
% Copyright (C) 2014 TNU, Institute for Biomedical Engineering, University of Zurich and ETH Zurich.
%
% This file is part of the TAPAS PhysIO Toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.



% template should only be length of a fraction of average heartbeat length   
defaults.shortenTemplateFactor = 0.5; 
defaults.minCycleSamples = ceil(0.5/2e-3);
defaults.thresh_min = 0.4;
defaults.doNormalizeTemplate = true;
% outliers below that correlation will be removed for averaging when retrieving final template
defaults.thresholdHighQualityCorrelation = 0.95; 

args = tapas_physio_propval(varargin, defaults);
tapas_physio_strip_fields(args);


doDebug = verbose.level >= 3;
dt = t(2) - t(1);

%% Guess peaks in two steps with updated average heartrate
% First step

[tmp, cpulse1stGuess] = tapas_physio_findpeaks( ...
    c,'minpeakheight',thresh_min,'minpeakdistance', minCycleSamples);

hasFirstGuessPeaks = ~isempty(cpulse1stGuess);


%% Second step, refined heart rate estimate

stringTitle = 'Preproc: Iterative Template Creation Single Cycle';
     
if hasFirstGuessPeaks
    
    averageHeartRateInSamples = round(mean(diff(cpulse1stGuess)));
    [tmp, cpulse2ndGuess] = tapas_physio_findpeaks(c,...
        'minpeakheight', thresh_min,...
        'minpeakdistance', round(shortenTemplateFactor*...
        averageHeartRateInSamples));
    
    if doDebug
        
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
    if doDebug
        fh = tapas_physio_get_default_fig_params();
        verbose.fig_handles(end+1) = fh;
        subplot(3,1,1);
        plot(t, c, 'k'); title('Preproc: Finding first peak of cycle, backwards')
    end
    verbose = tapas_physio_log(['No peaks found in raw cardiac time series. Check raw ' ...
        'physiological recordings figure whether there is any non-constant ' ...
        'cardiac data'], verbose, 2); % error!
end



%% Build template based on the guessed peaks:
% cut out all data around the detected (presumed) R-peaks
%   => these are the representative "QRS"-waves


halfTemplateWidthInSamples = round(shortenTemplateFactor * ...
    (averageHeartRateInSamples / 2));


[pulseCleanedTemplate, pulseTemplate] = tapas_physio_get_template_from_pulses(...
    c, cpulse2ndGuess, halfTemplateWidthInSamples, ...
    verbose, 'doNormalizeTemplate', doNormalizeTemplate, ...
    'thresholdHighQualityCorrelation', thresholdHighQualityCorrelation, ...
    'dt', dt);