function fh = tapas_physio_plot_raw_physdata(ons_secs)
% plots raw data from physiological logfile(s) to check whether read-in worked
%
%   fh = tapas_physio_plot_raw_physdata(ons_secs)
%
% IN
%   ons_secs    structure with the following fields
%   .t           [Nsamples, 1] time onsets vector (in seconds)
%   .c          [Nsamples, 1] cardiac (ECG or pulse oximetry) time series
%   .r          [Nsamples, 1] breathing (belt) amplitude time series
%   .cpulse     [Nheartbeats, 1] vector of onset times of heartbeats (R-peak) (in seconds)
%
% OUT
%   fh          figure handles where plot is drawn to
%
% EXAMPLE
%   tapas_physio_plot_raw_physdata
%
%   See also
%
% Author: Lars Kasper
% Created: 2013-02-21
% Copyright (C) 2013, Institute for Biomedical Engineering, ETH/Uni Zurich.
%
% This file is part of the PhysIO toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.
%
% $Id: tapas_physio_plot_raw_physdata.m 354 2013-12-02 22:21:41Z kasperla $
fh = tapas_physio_get_default_fig_params();
set(fh, 'Name', 'Raw Physiological Logfile Data');

has_cardiac = isfield(ons_secs, 'c') && ~isempty(ons_secs.c);
has_respiration = isfield(ons_secs, 'r') && ~isempty(ons_secs.r);
lg = cell(0,1);

if has_cardiac
    amp = max(ons_secs.c);
else
    if has_respiration
        amp = max(ons_secs.r);
    else
        amp = 1;
    end
end

if isfield(ons_secs, 'cpulse') && ~isempty(ons_secs.cpulse)
    stem(ons_secs.cpulse,amp*ones(size(ons_secs.cpulse)), 'm'); hold all;
    lg{end+1} = 'cardiac R-peak (heartbeat) events';
else
    warning('No cardiac R-peak (heartbeat) events provided');
end


if has_cardiac
    plot(ons_secs.t, ons_secs.c, 'r'); hold all;
    lg{end+1} = 'cardiac time course';
else
    warning('No cardiac time series provided');
end

if has_respiration
    plot(ons_secs.t, ons_secs.r, 'g'); hold all;
    lg{end+1} = 'respiratory time course';
else
    warning('No respiratory time series provided');
end

if ~isempty(lg), legend(lg); end;

title('Raw Physiological Logfile Data');
xlabel('t (s)');
