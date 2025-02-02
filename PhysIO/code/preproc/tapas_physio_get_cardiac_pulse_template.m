function [pulseCleanedTemplate, cpulse2ndGuess, averageHeartRateInSamples, verbose] = ...
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


if hasFirstGuessPeaks

    averageHeartRateInSamples = round(mean(diff(cpulse1stGuess)));
    [tmp, cpulse2ndGuess] = tapas_physio_findpeaks(c,...
        'minpeakheight', thresh_min,...
        'minpeakdistance', round(shortenTemplateFactor*...
        averageHeartRateInSamples));

    if doDebug

%% Second step, refined heart rate estimate

        nPulses1 = length(cpulse1stGuess);
        nPulses2 = length(cpulse2ndGuess);

        meanLag1 = mean(diff(t(cpulse1stGuess)));
        meanLag2 = mean(diff(t(cpulse2ndGuess)));

    end

else
    verbose = tapas_physio_log(['No peaks found in raw cardiac time series. Check raw ' ...
        'physiological recordings figure whether there is any non-constant ' ...
        'cardiac data'], verbose, 2); % error!
end

%% Plot in case of Debugging (verbose =>3)

if doDebug

    [verbose] = tapas_physio_plot_iterative_template_creation(hasFirstGuessPeaks,...
    t, c, cpulse1stGuess, nPulses1, nPulses2, cpulse2ndGuess, meanLag1, meanLag2, verbose);

    %save relevant functions
    verbose.review.iter_temp.hasFirstGuessPeaks = hasFirstGuessPeaks;
    verbose.review.iter_temp.t = t;
    verbose.review.iter_temp.c = c;
    verbose.review.iter_temp.cpulse1stGuess = cpulse1stGuess;
    verbose.review.iter_temp.nPulses1 = nPulses1;
    verbose.review.iter_temp.nPulses2 = nPulses2;
    verbose.review.iter_temp.cpulse2ndGuess = cpulse2ndGuess;
    verbose.review.iter_temp.meanLag1 = meanLag1; 
    verbose.review.iter_temp.meanLag2 = meanLag2;
end


%% Build template based on the guessed peaks:
% cut out all data around the detected (presumed) R-peaks
%   => these are the representative "QRS"-waves


halfTemplateWidthInSamples = round(shortenTemplateFactor * ...
    (averageHeartRateInSamples / 2));


[pulseCleanedTemplate, pulseTemplate, verbose] = tapas_physio_get_template_from_pulses(...
    c, cpulse2ndGuess, halfTemplateWidthInSamples, ...
    verbose, 'doNormalizeTemplate', doNormalizeTemplate, ...
    'thresholdHighQualityCorrelation', thresholdHighQualityCorrelation, ...
    'dt', dt);