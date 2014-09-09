function [c, r, t, cpulse] = physio_read_physlogfiles_GE(files)
% reads out physiological time series and timing vector depending on the
% MR scanner vendor and the modality of peripheral cardiac monitoring (ECG
% or pulse oximetry)
%
%   [cpulse, rpulse, t, c] = physio_read_physlogfiles_GE(logfile, vendor, cardiac_modality)
%
% IN    files
%       .log_cardiac        contains ECG or pulse oximeter time course
%                           for GE: ECGData...
%       .log_respiration    contains breathing belt amplitude time course
%                           for GE: RespData...
%                           
% OUT
%   cpulse              time events of R-wave peak in cardiac time series (seconds)
%                       for GE: usually empty
%   r                   respiratory time series
%   t                   vector of time points (in seconds)
%                       NOTE: This assumes the default sampling rate of 40
%                       Hz
%   c                   cardiac time series (ECG or pulse oximetry)
%
% EXAMPLE
%   [ons_secs.cpulse, ons_secs.rpulse, ons_secs.t, ons_secs.c] =
%       physio_read_physlogfiles_GE(logfile, vendor, cardiac_modality);
%
%   See also physio_main_create_regressors
%
% Author: Lars Kasper
% Created: 2013-02-16
% Copyright (C) 2013 Institute for Biomedical Engineering, ETH/Uni Zurich.
%
% This file is part of the PhysIO toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.
%
% $Id: physio_read_physlogfiles_GE.m 183 2013-04-25 15:55:05Z kasperla $

%% read out values

dt = 25/1000;

c = load(files.cardiac);
r = load(files.respiration);
Nsamples = size(c,1);
t =((0:(Nsamples-1))*dt)'; 
cpulse = [];

end
