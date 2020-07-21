function verbose = tapas_physio_plot_raw_physdata(ons_secs, verbose)
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

% Author: Lars Kasper
% Created: 2013-02-21
% Copyright (C) 2013, Institute for Biomedical Engineering, ETH/Uni Zurich.
%
% This file is part of the PhysIO toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

if verbose.level >= 2

    fh = tapas_physio_get_default_fig_params(verbose);

    set(fh, 'Name', 'Read-In: Raw Physiological Logfile Data');
    has_cardiac_triggers = isfield(ons_secs, 'cpulse') && ~isempty(ons_secs.cpulse);
    has_scan_triggers = isfield(ons_secs, 'acq_codes') && ~isempty(ons_secs.acq_codes);
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
    
    
    if has_scan_triggers
        plot(ons_secs.t, amp*ons_secs.acq_codes/max(ons_secs.acq_codes), 'c'); hold all;
        lg{end+1} = 'Scan trigger events';
    else
        verbose = tapas_physio_log('No scan trigger events provided', verbose, 0);
    end
    

    if has_cardiac_triggers
        stem(ons_secs.cpulse,amp*ones(size(ons_secs.cpulse)), 'm'); hold all;
        lg{end+1} = 'Cardiac R-peak (heartbeat) events';
    else
        verbose = tapas_physio_log('No cardiac R-peak (heartbeat) events provided', verbose, 0);
    end
    
    
    if has_cardiac
        plot(ons_secs.t, ons_secs.c, 'r'); hold all;
        lg{end+1} = 'Cardiac time course';
    else
        verbose = tapas_physio_log('No cardiac time series provided', verbose, 1);
    end
    
    if has_respiration
        plot(ons_secs.t, ons_secs.r, 'g'); hold all;
        lg{end+1} = 'Respiratory time course';
    else
        verbose = tapas_physio_log('No respiratory time series provided', verbose, 1);
    end
    
    if ~isempty(lg), legend(lg); end;
    
    title('Raw Physiological Logfile Data');
    xlabel(sprintf('t (s) (relative to t_{start} = %.2f s)', ons_secs.t_start));
    
    verbose.fig_handles(end+1) = fh;
    
end