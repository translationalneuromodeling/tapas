function [cardiac_phase, verbose] = tapas_physio_get_cardiac_phase(...
    pulset,scannert, verbose, svolpulse)
% estimates cardiac phases from cardiac pulse data
%
% USAGE
%   cardiac_phase = tapas_physio_get_cardiac_phase(pulset,scannert, verbose, svolpulse)
%
% INPUT
%        pulset     - heart-beat/pulseoxymeter data read from spike file
%        scannert   - scanner slice pulses read from log file
%        verbose    - set verbose.level >=3, if figures for debugging are wanted
%        svolpulse  - volume start pulses from log file (only for plot reasons)
%
% OUTPUT
%        cardiac_phase - phase in heart-cycle when each slice of each
%        volume was acquired
%        fh         - figure handle
%
% The regressors are calculated as described in
% Glover et al, 2000, MRM, (44) 162-167
% Josephs et al, 1997, ISMRM, p1682
%_______________________________________________________________________
% Author: Lars Kasper, heavily based on an earlier implementation of
%                      Eric Featherstone and Chloe Hutton 26/03/07 (FIL,
%                      UCL London)
%
% Copyright (C) 2009, Institute for Biomedical Engineering, ETH/Uni Zurich.
%
% This file is part of the PhysIO toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

%


% Find the time of pulses just before and just after each scanner time
% point. Where points are missing, fill with NaNs and set to zero later.

isVerbose = verbose.level >=3;
nPulses = numel(pulset);

% number of cycles to average for heart beat duration guess
nAverage = min(20, nPulses/2);

% add pulses in beginning, if first onset time of scan before recorded
% pulses - use guess based on average heart rate
if scannert(1) <= pulset(1)
    verbose = tapas_physio_log(...
        'Guessed additional cardiac pulse at time series start for phase estimation', ...
        verbose, 1);
    meanCycleDur = mean(diff(pulset(1:nAverage)));
    nAddPulses = max(1, ceil((pulset(1) - scannert(1))/meanCycleDur));
    pulset = [pulset(1) - meanCycleDur*(1:nAddPulses)';pulset];
end

% add pulses in the end, if last onset time of scan after last recorded
% pulses - use guess based on average heart rate
if scannert(end) > pulset(end)
    % add more pulses before first one, using same average heartbeat
    % duration as in first nAverage cycles
    meanCycleDur = mean(diff(pulset((end-nAverage+1):end)));
    nAddPulses = ceil((scannert(end) - pulset(end))/meanCycleDur);
    pulset = [pulset; pulset(end) + meanCycleDur*(1:nAddPulses)'];
    verbose = tapas_physio_log(...
        sprintf('Note: Guessed %d additional cardiac pulse(s) at time series end for phase estimation', nAddPulses), ...
        verbose, 0);
end

scannertpriorpulse = zeros(1,length(scannert));
scannertafterpulse = scannertpriorpulse;
for i=1:length(scannert)
    % check for prior heartbeat
    n = find(pulset < scannert(i), 1, 'last');
    scannertpriorpulse(i) = pulset(n);
    scannertafterpulse(i) = pulset(n+1);
end

% Calculate cardiac phase at each slice (from Glover et al, 2000).
cardiac_phase=(2*pi*(scannert'-scannertpriorpulse)./(scannertafterpulse-scannertpriorpulse))';

if isVerbose
    % 1. plot chosen slice start event
    % 2. plot chosen c_sample phase on top of chosen slice scan start, (as a stem
    % and line of phases)
    % 3. plot all detected cardiac r-wave peaks
    % 4. plot volume start event
    stringTitle = 'Preproc: tapas_physio_get_cardiac_phase: scanner and R-wave pulses - output phase';
    fh = tapas_physio_get_default_fig_params();
    set(fh, 'Name', stringTitle);
    stem(scannert, cardiac_phase, 'k'); hold on;
    plot(scannert, cardiac_phase, 'k');
    stem(pulset,3*ones(size(pulset)),'r', 'LineWidth',2);
    stem(svolpulse,7*ones(size(svolpulse)),'g', 'LineWidth',2);
    legend('estimated phase at slice events', ...
        '', ...
        'heart beat R-peak', ...
        'scan volume start');
    title(regexprep(stringTitle,'_', '\\_'));
    xlabel('t (seconds)');
    %stem(scannertpriorpulse,ones(size(scannertpriorpulse))*2,'g');
    %stem(scannertafterpulse,ones(size(scannertafterpulse))*2,'b');
end