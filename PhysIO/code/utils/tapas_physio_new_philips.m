function physio = tapas_physio_new_philips(physio)
% Initializes physio-properties for Philips 3T Achieva system with good
% ECG-data
%
%   physio = tapas_physio_new_philips(physio)
%
% IN
%
% OUT
%
% EXAMPLE
%   tapas_physio_new_philips
%
%   See also

% Author: Lars Kasper
% Created: 2014-10-09
% Copyright (C) 2014 TNU, Institute for Biomedical Engineering, University of Zurich and ETH Zurich.
%
% This file is part of the TAPAS PhysIO Toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.

log_files   = physio.log_files;
scan_timing = physio.scan_timing;
model       = physio.model;
preproc     = physio.preproc;
verbose     = physio.verbose;

sqpar       = scan_timing.sqpar;
sync        = scan_timing.sync;

model.type = 'RETROICOR';
model.order = struct('c',3,'r',4,'cr',1, 'orthogonalise', 'none');
log_files.sampling_interval = 1/496; % for WiFi 1/496, cable: 1/500

sqpar.Ndummies = 0; % PARAM
sqpar.Nscans = 160; % PARAM
sqpar.Nprep = [];
sqpar.time_slice_to_slice = [];

model.order.c = 3;
model.order.r = 4;
model.order.cr = 1;
model.type = 'RETROICOR';

verbose.level = 2; % PARAM
verbose.fig_output_file ='physio_output.png';
verbose.use_tabs = 0;

sync.method = 'gradient_log';
sync.grad_direction = 'z';
sync.slice = 1500;
sync.zero = 1400;
sync.vol_spacing = [];

preproc.cardiac.modality = 'ECG';
preproc.cardiac.initial_cpulse_select.method = 'load_from_logfile'; 'auto_matched';
preproc.cardiac.posthoc_cpulse_select.method = 'off';

scan_timing.sqpar = sqpar;
scan_timing.sync = sync;

physio.log_files = log_files;
physio.scan_timing = scan_timing;
physio.preproc = preproc;
physio.model = model;
physio.verbose = verbose;
