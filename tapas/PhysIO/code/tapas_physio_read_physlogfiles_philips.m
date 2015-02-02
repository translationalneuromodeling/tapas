function [c, r, t, cpulse, acq_codes] = tapas_physio_read_physlogfiles_philips(log_files, cardiac_modality)
% reads out physiological time series and timing vector depending on the
% MR scanner vendor and the modality of peripheral cardiac monitoring (ECG
% or pulse oximetry)
%
%   [c, r, t, cpulse] = tapas_physio_read_physlogfiles_philips(logfile, vendor, cardiac_modality)
%
% IN
%   log_files                   tapas.log_files; see also tapas_physio_new
%           .respiratory
%           .cardiac
%           .sampling_interval
%           .relative_start_acquisition
%   cardiac_modality    'ECG' for ECG, 'OXY'/'PPU' for pulse oximetry
%                       Note: Append '_wifi'
%                       for adjusted sampling rate in
%                       wireless Ingenia Scanners
%
% OUT
%   c                   cardiac time series (ECG or pulse oximetry)
%   r                   respiratory time series
%   t                   vector of time points (in seconds)
%   cpulse              time events of R-wave peak in cardiac time series (seconds)
%   acq_codes           slice/volume start events marked by number <> 0
%                       for time points in t
%
% EXAMPLE
%   [ons_secs.cpulse, ons_secs.rpulse, ons_secs.t, ons_secs.c] =
%   tapas_physio_read_physlogfiles(logfile, vendor, cardiac_modality);
%
%   See also tapas_physio_main_create_regressors
%
% Author: Lars Kasper
% Created: 2013-02-16
% Copyright (C) 2013, Institute for Biomedical Engineering, ETH/Uni Zurich.
%
% This file is part of the PhysIO toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.
%
% $Id: tapas_physio_read_physlogfiles_philips.m 640 2015-01-11 22:03:32Z kasperla $

%% read out values
hasCardiac  = ~isempty(log_files.cardiac);
hasResp     = ~isempty(log_files.respiration);

if hasCardiac
    logfile = log_files.cardiac;
else
    logfile = log_files.respiration;
end

y = tapas_physio_read_physlogfiles_philips_matrix(logfile);

acq_codes   = y(:,10);

Nsamples    = size(y,1);

dt          = log_files.sampling_interval;

%default: 500 Hz sampling frequency
isWifi      = regexpi(cardiac_modality, '_wifi');

if isWifi
    cardiac_modality = regexprep(cardiac_modality, '_wifi', '', 'ignorecase');
end

if isempty(dt)
    if isWifi
        dt = 1/496;
    else
        dt = 2e-3;
    end
end

t= -log_files.relative_start_acquisition + ((0:(Nsamples-1))*dt)';



% column 3 = ECG, 5 = PPU, 6 = resp,
% 10 = scanner signal: 10/20 = scan start/end; 1 = ECG pulse; 2 = OXY max; 8 = scan event TODO: what is 3 and 9???
% columns 7,8,9: Grad-strengh x,y,z

cpulse = find(acq_codes==1);
if ~isempty(cpulse)
    cpulse = t(cpulse);
end;

if hasResp
    r = y(:,6);
else
    r = [];
end


if hasCardiac
    
    switch lower(cardiac_modality)
        case {'ecg', 'ecg_filtered'}
            c = y(:,3);
        case {'ecg_raw'}
            c = y(:,1);
        case {'oxy','oxyge', 'ppu'}
            c = y(:,5);
    end
else
    c = [];
end

end
