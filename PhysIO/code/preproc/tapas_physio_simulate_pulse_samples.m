function simulatedSamples = tapas_physio_simulate_pulse_samples(t, c, ...
    nSimulatedSamples, positionString, verbose)
% Simulates samples at start/end of physiological recording by estimating
% pulse template and average pulse rate, and continuing those
%
% simulatedSamples = tapas_physio_simulate_pulse_samples(t, c, ...
%     nSimulatedSamples, positionString, verbose)%
% IN
%
% OUT
%
% EXAMPLE
%   tapas_physio_simulate_pulse_samples
%
%   See also tapas_physio_get_cardiac_pulse_template tapas_physio_read_physlogfiles

% Author:   Lars Kasper
% Created:  2019-01-26
% Copyright (C) 2019 TNU, Institute for Biomedical Engineering,
%                    University of Zurich and ETH Zurich.
%
% This file is part of the TAPAS PhysIO Toolbox, which is released under
% the terms of the GNU General Public License (GPL), version 3. You can
% redistribute it and/or modify it under the terms of the GPL (either
% version 3 or, at your option, any later version). For further details,
% see the file COPYING or <http://www.gnu.org/licenses/>.

doDebug = verbose.level >=2;

% add artificial time series before
simulatedPulses = zeros(nSimulatedSamples, 1);
[pulseTemplate, idxPulses, meanPulseRateInSamples] = ...
    tapas_physio_get_cardiac_pulse_template(t, c, verbose, ...
    'doNormalizeTemplate', false, 'shortenTemplateFactor', .99);

doPrepend = false;
doAppend = false;
switch positionString
    case {'pre', 'before'}
        doPrepend = true;
        % Create last simulated pulse:
        % if first detected pulse in orig time series is further from start
        % than meanPulseRate, assume a regular heart beat also all the way
        % to the start, i.e. spacing by a multiple of meanPulseRateInSamples
        % to last simulated pulse
        if mod(idxPulses(1), meanPulseRateInSamples) == 0
            % if earliest pulse at multiple of pulse rate, 
            % then next pulse would be just before start of orig time
            % series
            idxLastSimulatedPulse = nSimulatedSamples;
        else 
            idxLastSimulatedPulse = nSimulatedSamples - meanPulseRateInSamples ...
                + mod(idxPulses(1), meanPulseRateInSamples);
        end
        % put 1 where pulses should occur before start of time series,
        % given earliest pulse and mean pulse rate, fill backwards...
        simulatedPulses(idxLastSimulatedPulse:-meanPulseRateInSamples:1) = 1;
    case {'post', 'after'}
        doAppend = true;
        % Create first simulated pulse:
        % if last detected pulse in orig time series is further from end
        % than meanPulseRate, assume a regular heart beat also all the way
        % to the end, i.e. spacing by a multiple of meanPulseRateInSamples
        % to first simulated pulse
        nSamplesOrig = numel(c);
        idxFirstSimulatedPulse = meanPulseRateInSamples ...
           - mod(nSamplesOrig - idxPulses(end), meanPulseRateInSamples);
        
        % put 1 where pulses should occur after end of time series
        % given last pulse event in orig time series and mean pulse rate...
        simulatedPulses(idxFirstSimulatedPulse:meanPulseRateInSamples:end) = 1;
    otherwise
        tapas_physio_log(...
            sprintf('Unknown positionString ''%s'' for simulating samples; Use ''pre'' or ''post''', positionString),...
            verbose, 2)
end

% ...and convolve with pulse template
% **TODO** tapas_physio_conv
simulatedSamples = conv(simulatedPulses, pulseTemplate, 'same');

if doDebug
    dt = t(2) - t(1);
    if doPrepend
        tNew = -flip(1:nSimulatedSamples)'*dt+t(1);
    elseif doAppend
        tNew = (1:nSimulatedSamples)'*dt+t(end);
    end
    verbose.fig_handles(end+1) = tapas_physio_get_default_fig_params();
    stringTitle = sprintf('Preproc: Added simulated samples %s time series', positionString);
    
    % plot orig time series and extension by simulated samples
    plot(t,c); hold all;
    plot(tNew, simulatedSamples);
    
    % plot time events of actual and simulated pulses as stems
    idxSimulatedPulses = find(simulatedPulses);
    stem(t(idxPulses), ones(numel(idxPulses)));
    stem(tNew(idxSimulatedPulses), ones(numel(idxSimulatedPulses)));
    
    % plot template pulse centered on one simulated pulse
    tStart = tNew(idxSimulatedPulses(1));
    nSamplesTemplate = numel(pulseTemplate);
    tTemplate = tStart+dt*(-ceil(pulseTemplate/2)+(0:(nSamplesTemplate-1)))';
    plot(tTemplate, pulseTemplate);
    title(stringTitle);
end