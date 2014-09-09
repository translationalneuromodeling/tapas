function [c, r, t, cpulse] = tapas_physio_read_physlogfiles_custom(log_files)
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
% $Id: tapas_physio_read_physlogfiles_custom.m 466 2014-04-27 13:10:48Z kasperla $

%% read out values
if ~isempty(log_files.respiration)
    r = load(log_files.respiration, 'ascii');
else 
    r = [];
end

if ~isempty(log_files.cardiac)
    c = load(log_files.cardiac, 'ascii');
else 
    c = [];
end
nSamples = max(size(c,1), size(r,1));



dt = log_files.sampling_interval; %500 Hz sampling frequency
t= -log_files.relative_start_acquisition + ((0:(nSamples-1))*dt)'; 

hasCpulses = size(c,2) > 1; %2nd column with pulse indicator set to one

if hasCpulses
    cpulse = find(c(:,2)==1);
    cpulse = t(cpulse);
    c = c(:,1);
else
    cpulse = [];
end
