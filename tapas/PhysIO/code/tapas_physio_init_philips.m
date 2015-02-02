function physio = tapas_physio_init_philips(physio)
% Initializes physio-properties for Philips 3T Achieva system with good
% ECG-data
%
%   physio = tapas_physio_init_philips(physio)
%
% IN
%
% OUT
%
% EXAMPLE
%   tapas_physio_init_philips
%
%   See also
%
% Author: Lars Kasper
% Created: 2014-10-09
% Copyright (C) 2014 TNU, Institute for Biomedical Engineering, University of Zurich and ETH Zurich.
%
% This file is part of the TAPAS PhysIO Toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.
%
% $Id: tapas_physio_init_philips.m 539 2014-10-09 15:59:25Z kasperla $

log_files = physio.log_files;
sqpar = physio.sqpar;
model = physio.model;
thresh = physio.thresh;
verbose = physio.verbose;


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

thresh.scan_timing.method = 'gradient_log';
thresh.scan_timing.grad_direction = 'z';
thresh.scan_timing.slice = 1500;
thresh.scan_timing.zero = 1400;
thresh.scan_timing.vol_spacing = [];

thresh.cardiac.modality = 'ECG';
thresh.cardiac.initial_cpulse_select.method = 'load_from_logfile'; 'auto_matched';
thresh.cardiac.posthoc_cpulse_select.method = 'off';

physio.log_files = log_files;
physio.sqpar = sqpar;
physio.model = model;
physio.thresh = thresh;
physio.verbose = verbose;
