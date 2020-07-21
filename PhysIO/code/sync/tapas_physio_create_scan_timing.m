function  [ons_secs, VOLLOCS, LOCS, verbose] = tapas_physio_create_scan_timing(...
    log_files, scan_timing, ons_secs, verbose)
% Extracts slice and volume scan onsets (triggers) from different vendor formats
%
%   [ons_secs, VOLLOCS, LOCS, verbose] = tapas_physio_create_scan_timing(...
%            log_files, scan_timing, ons_secs, verbose);
%
%
% IN
%   NOTE: The detailed description of all input structures can be found as
%   comments in tapas_physio_new
%
%   log_files    - file names (physiology and scan timing) and sampling rates
%
%   ons_secs     -  structure for time-dependent variables, i.e. onsets,
%                   specified in seconds, in particular
%                   .t          - time vector of phys time course
%
%   scan_timing         -  Parameters for sequence timing & synchronization
%   scan_tming.sqpar    -  slice and volume acquisition starts, TR,
%                          number of scans etc.
%   scan_timing.sync    -  synchronization options
%                          (e.g. from gradients, trigger, tics phys
%                           logfile to scan acquisition)
%
%       sqpar           - sequence timing parameters, used for computation
%                         of scan events from 'nominal' timing
%           .Nslices        - number of slices per volume in fMRI scan
%           .TR             - repetition time in seconds
%           .Ndummies       - number of dummy volumes
%           .Nscans         - number of full volumes saved (volumes in nifti file,
%                             usually rows in your design matrix)
%           .Nprep          - number of non-dummy, volume like preparation pulses
%                             before 1st dummy scan. If set, logfile is read from beginning,
%                             otherwise volumes are counted from last detected volume in the logfile
%           .time_slice_to_slice - time between the acquisition of 2 subsequent
%                             slices; typically TR/Nslices or
%                             minTR/Nslices, if minimal temporal slice
%                             spacing was chosen
%
%   verbose                 - defines output level (which graphics to plot
%                             and whether to save them)
%
% OUT
%   ons_secs    -  structure for time-dependent variables, i.e. onsets,
%                  specified in seconds, updated fields
%                   .spulse     - scan slice trigger events
%                   .svolpulse  - scan volume trigger events
%                   .spulse_per_vol
%                               - cell(nVolumes,1) of slice triggers per
%                                 volume
%                   .acq_codes  - acquisition codes (e.g. triggers) within
%                                 phys log files (e.g. Philips, Biopac)
%
%   VOLLOCS     - index locations in time vector (of physiological recordings),
%                             when volume scan events started
%   LOCS        - locations in time vector, when slice or volume scan
%                             events started
%
%   See also tapas_physio_new tapas_physio_main_create_regresssors

% Author: Lars Kasper
% Created: 2013-08-23
% Copyright (C) 2016 Institute for Biomedical Engineering, ETH/Uni Zurich.
%
% This file is part of the PhysIO toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.


sqpar   = scan_timing.sqpar;

% TODO: introduce auto that takes time stamps from default locations
% for different vendors
switch lower(scan_timing.sync.method)
    case 'nominal'
        [VOLLOCS, LOCS] = ...
            tapas_physio_create_scan_timing_nominal(ons_secs.t + ...
                    ons_secs.t_start, sqpar, log_files.align_scan);
    case {'gradient', 'gradient_log'}
        [VOLLOCS, LOCS, verbose] = ...
            tapas_physio_create_scan_timing_from_gradients_philips( ...
            log_files, scan_timing, verbose);
    case {'gradient_auto', 'gradient_log_auto'}
        [VOLLOCS, LOCS, verbose] = ...
            tapas_physio_create_scan_timing_from_gradients_auto_philips( ...
            log_files, scan_timing, verbose);
    case 'scan_timing_log'
        switch lower(log_files.vendor)
            case 'siemens'
                % for alignScan = 'last', in case logfile lasts longer than end of last scan
                % assuming t = 0 is already start of first scan
                durationPhyslogAfterEndOfLastScan = ons_secs.t(end) + ...
                    ons_secs.t_start - sqpar.Nscans*sqpar.TR;
                [VOLLOCS, LOCS] = ...
                    tapas_physio_create_scan_timing_nominal(ons_secs.t + ...
                    ons_secs.t_start, sqpar, log_files.align_scan, ...
                    durationPhyslogAfterEndOfLastScan);
            case 'siemens_tics'
                [VOLLOCS, LOCS, verbose] = ...
                    tapas_physio_create_scan_timing_from_tics_siemens( ...
                    ons_secs.t, ons_secs.t_start, log_files, verbose);
            case {'biopac_mat', 'biopac_txt', 'bids'}
                [VOLLOCS, LOCS, verbose] = ...
                    tapas_physio_create_scan_timing_from_acq_codes( ...
                    ons_secs.t + ons_secs.t_start, ons_secs.acq_codes, ...
                    sqpar, log_files.align_scan, verbose);
        end
    otherwise
        verbose = tapas_physio_log(...
            sprintf('unknown scan_timing.sync.method: %s', ...
            scan_timing.sync.method), verbose, 2);
end


% remove arbitrary offset in time vector now, since all timings have now
% been aligned to ons_secs.t
% ons_secs.t = ons_secs.t - ons_secs.t(1);

[ons_secs.svolpulse, ons_secs.spulse, ons_secs.spulse_per_vol, verbose] = ...
    tapas_physio_get_onsets_from_locs(...
    ons_secs.t, VOLLOCS, LOCS, sqpar, verbose);
