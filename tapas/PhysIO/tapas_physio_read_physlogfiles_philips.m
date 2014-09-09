function [c, r, t, cpulse] = tapas_physio_read_physlogfiles_philips(logfile, cardiac_modality)
% reads out physiological time series and timing vector depending on the
% MR scanner vendor and the modality of peripheral cardiac monitoring (ECG
% or pulse oximetry)
%
%   [cpulse, rpulse, t, c] = tapas_physio_read_physlogfiles_philips(logfile, vendor, cardiac_modality)
%
% IN
%   logfile             Philips scanphyslogfile (including path): 'SCANPHYSLOG_XXXXXXX.log';
%   cardiac_modality    'ECG' for ECG, 'OXY' for pulse oximetry
%
% OUT
%   cpulse              time events of R-wave peak in cardiac time series (seconds)
%   rpulse              respiratory time series
%   t                   vector of time points (in seconds)
%   c                   cardiac time series (ECG or pulse oximetry)
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
% $Id: tapas_physio_read_physlogfiles_philips.m 235 2013-08-19 16:28:07Z kasperla $

%% read out values
[z{1:10}]=textread(logfile,'%d %d %d %d %d %d %d %d %d %d','commentstyle', 'shell');
y = cell2mat(z);

Nsamples=size(y,1);

dt = 2e-3; %500 Hz sampling frequency
t=((0:(Nsamples-1))*dt)'; 



% column 3 = ECG, 5 = PPU, 6 = resp, 
% 10 = scanner signal: 10/20 = scan start/end; 1 = ECG pulse; 2 = OXY max; 8 = scan event TODO: what is 3 and 9???
% columns 7,8,9: Grad-strengh x,y,z

cpulse = find(z{10}==1);
if ~isempty(cpulse)
    cpulse = t(cpulse);
end;

r = y(:,6);

switch lower(cardiac_modality)
    case {'ecg', 'ecg_filtered'}
        c = y(:,3);
    case {'ecg_raw'}
        c = y(:,1);
    case {'oxy','oxyge'}
        c = y(:,5);
end

end
