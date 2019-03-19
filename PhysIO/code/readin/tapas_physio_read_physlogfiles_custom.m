function [c, r, t, cpulse] = tapas_physio_read_physlogfiles_custom(log_files, ...
    verbose)
% reads out physiological time series and timing vector from custom-made logfiles
%   of peripheral cardiac monitoring (ECG
% or pulse oximetry)
%
%    [c, r, t, cpulse] = tapas_physio_read_physlogfiles_custom(logfiles)
%
% IN
%   log_files                   tapas.log_files; see also tapas_physio_new
%           .respiratory
%           .cardiac
%           .sampling_interval
%           .relative_start_acquisition
% OUT
%   c                   cardiac time series (ECG or pulse oximetry)
%   r                   respiratory time series
%   t                   vector of time points (in seconds)
%   cpulse              time events of R-wave peak in cardiac time series (seconds)
%
% EXAMPLE
%   [ons_secs.cpulse, ons_secs.rpulse, ons_secs.t, ons_secs.c] =
%   tapas_physio_read_physlogfiles(logfile, vendor, cardiac_modality);
%
%   See also tapas_physio_main_create_regressors

% Author: Lars Kasper
% Created: 2013-02-16
% Copyright (C) 2013, Institute for Biomedical Engineering, ETH/Uni Zurich.
%
% This file is part of the PhysIO toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.


%% read out values
DEBUG = verbose.level >=3;

hasRespirationFile = ~isempty(log_files.respiration);
hasCardiacFile = ~isempty(log_files.cardiac);

if hasRespirationFile
    r = load(log_files.respiration, 'ascii');
else 
    r = [];
end

if hasCardiacFile
    c = load(log_files.cardiac, 'ascii');
else 
    c = [];
end

%% resample data, if differen sampling frequencies given
dt = log_files.sampling_interval;

hasDifferentSamplingRates = numel(dt) > 1;

if hasDifferentSamplingRates && hasCardiacFile && hasRespirationFile
    dtCardiac = dt(1);
    dtRespiration = dt(2);
    isHigherSamplingCardiac = dtCardiac < dtRespiration;
    
    nSamplesRespiration = size(r,1);
    nSamplesCardiac = size(c,1);
    
    tCardiac = -log_files.relative_start_acquisition + ...
        ((0:(nSamplesCardiac-1))*dtCardiac)';
    
    
    tRespiration = -log_files.relative_start_acquisition + ...
        ((0:(nSamplesRespiration-1))*dtRespiration)';
    
    if isHigherSamplingCardiac
        t = tCardiac;
        rInterp = interp1(tRespiration, r, t);
        
        if DEBUG
            fh = plot_interpolation(tRespiration, r, t, rInterp, ...
                {'respiratory', 'cardiac'});
            verbose.fig_handles(end+1) = fh;
        end
        r = rInterp;
        
    else
        t = tRespiration;
        cInterp = interp1(tCardiac, c, t);
        
        if DEBUG
            fh = plot_interpolation(tCardiac, c, t, cInterp, ...
                {'cardiac', 'respiratory'});
            verbose.fig_handles(end+1) = fh;
        end
        c = cInterp;
          
    end
    
else
    nSamples = max(size(c,1), size(r,1));
    t = -log_files.relative_start_acquisition + ((0:(nSamples-1))*dt)';
end

hasCpulses = size(c,2) > 1; %2nd column with pulse indicator set to one

if hasCpulses
    cpulse = find(c(:,2)==1);
    cpulse = t(cpulse);
    c = c(:,1);
else
    cpulse = [];
end

end

%% local function to plot interpolation result
function fh = plot_interpolation(tOrig, yOrig, tInterp, yInterp, ...
    stringOrigInterp)
fh = tapas_physio_get_default_fig_params;
stringTitle = sprintf('Read-In: Interpolation of %s signal', stringOrigInterp{1});
set(fh, 'Name', stringTitle);
plot(tOrig, yOrig, 'go--');  hold all;
plot(tInterp, yInterp,'r.');
xlabel('t (seconds');
legend({
    sprintf('after interpolation to %s timing', ...
    stringOrigInterp{1}), ...
    sprintf('original %s time series', stringOrigInterp{2}) });
title(stringTitle);
end
