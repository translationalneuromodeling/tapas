function [c, r, t, cpulse, acq_codes, verbose] = tapas_physio_read_physlogfiles(log_files, cardiac_modality, ...
    verbose, sqpar)
% reads out physiological time series and timing vector depending on the
% MR scanner vendor and the modality of peripheral cardiac monitoring (ECG
% or pulse oximetry)
%
% [c, r, t, cpulse, acq_codes, verbose] = tapas_physio_read_physlogfiles(log_files, cardiac_modality, ...
%    verbose)
%
% IN
%   log_files   is a structure containing the following filenames (with full
%           path)
%       .vendor             'Philips', 'GE' or 'Siemens', depending on your
%                           MR Scanner system
%       .log_cardiac        contains ECG or pulse oximeter time course
%                           for Philips: 'SCANPHYSLOG<DATE&TIME>.log';
%                           can be found on scanner in G:/log/scanphyslog-
%                           directory, one file is created per scan, make sure to take
%                           the one with the time stamp corresponding to your PAR/REC
%                           files
%       .log_respiration    contains breathing belt amplitude time course
%                           for Philips: same as .log_cardiac
%   cardiac_modality    'ECG' for ECG, 'OXY'/'PPU' for pulse oximetry, default: 'ECG'
%   sqpar                   sequence parameters (TR, nScans etc.)
%                           only needed for time adjustments of logfile
%                           relative to duration of a selected run
% OUT
%   c                   cardiac time series (ECG or pulse oximetry)
%   r                   respiratory time series
%   t                   vector of time points (in seconds)
%   cpulse              time events of R-wave peak in cardiac time series (seconds)
%   acq_codes           slice/volume start events marked by number <> 0
%                       for time points in t
%                       10/20 = scan start/end;
%                       1 = ECG pulse; 2 = OXY max; 3 = Resp trigger;
%                       8 = scan volume trigger
%
% EXAMPLE
%   [ons_secs.cpulse, ons_secs.rpulse, ons_secs.t, ons_secs.c] =
%   tapas_physio_read_physlogfiles(logfile, vendor, cardiac_modality);
%
%   See physio_also main_create_regressors

% Author: Lars Kasper
% Created: 2013-02-16
% Copyright (C) 2013, Institute for Biomedical Engineering, ETH/Uni Zurich.
%
% This file is part of the PhysIO toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

if nargin < 2
    cardiac_modality = 'ECG';
end

if nargin < 3
    verbose.level = 0;
end

switch lower(log_files.vendor)
    case 'bids'
        [c, r, t, cpulse, acq_codes] = ...
            tapas_physio_read_physlogfiles_bids(log_files, cardiac_modality, verbose);
    case 'biopac_mat'
        [c, r, t, cpulse, acq_codes] = ...
            tapas_physio_read_physlogfiles_biopac_mat(log_files, cardiac_modality, verbose);
    case 'biopac_txt'
        [c, r, t, cpulse, acq_codes, verbose, gsr] = ...
            tapas_physio_read_physlogfiles_biopac_txt(log_files, cardiac_modality, verbose);
    case 'custom'
        [c, r, t, cpulse] = ...
            tapas_physio_read_physlogfiles_custom(log_files, verbose);
        acq_codes = [];
    case 'brainproducts'
        [c, r, t, cpulse, acq_codes] = ...
            tapas_physio_read_physlogfiles_brainproducts(log_files, cardiac_modality, verbose);
    case 'ge'
        [c, r, t, cpulse] = ...
            tapas_physio_read_physlogfiles_GE(log_files, verbose);
        acq_codes = [];
    case 'philips'
        [c, r, t, cpulse, acq_codes] = ...
            tapas_physio_read_physlogfiles_philips(log_files, cardiac_modality);
    case 'siemens'
        [c, r, t, cpulse, verbose] = ...
            tapas_physio_read_physlogfiles_siemens(log_files, cardiac_modality, verbose, ...
            'sqpar', sqpar);
        acq_codes = [];
    case 'siemens_tics'
        [c, r, t, cpulse, acq_codes, verbose] = ...
            tapas_physio_read_physlogfiles_siemens_tics(log_files, cardiac_modality, verbose);
    case 'siemens_hcp'
        [c, r, t, cpulse, acq_codes, verbose] = ...
            tapas_physio_read_physlogfiles_siemens_hcp(log_files, cardiac_modality, verbose);
end

% Do not prepend for Siemens Tics, since can be as long as a day
isSiemensTics = strcmpi(log_files.vendor, 'siemens_tics');

% prepend all data with zeros for better processing, if scan starts before
% physiological data
if ~isempty(t) && t(1) > 0 && ~isSiemensTics
    dt = t(2) - t(1);
    nPrependSamples = ceil(t(1)/dt);
    if ~isempty(c)
        prependSamples = tapas_physio_simulate_pulse_samples(t, c, nPrependSamples, 'pre', verbose);
        c = [prependSamples;c];
    end
    if ~isempty(r)
        prependSamples = tapas_physio_simulate_pulse_samples(t, r, nPrependSamples, 'pre', verbose);
        r = [prependSamples;r];
    end
    if ~isempty(acq_codes)
        acq_codes = [zeros(nPrependSamples,1);acq_codes];
    end
    
    t = [(0:nPrependSamples-1)'*dt;t]; 
end
end
