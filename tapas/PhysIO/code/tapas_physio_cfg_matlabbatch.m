function physio = tapas_physio_cfg_matlabbatch
% Lars Kasper, March 2013
%
% Copyright (C) 2013, Institute for Biomedical Engineering, ETH/Uni Zurich.
%
% This file is part of the TAPAS PhysIO Toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.
%
% $Id: tapas_physio_cfg_matlabbatch.m 668 2015-02-01 12:22:26Z kasperla $


pathThis = fileparts(mfilename('fullpath')); % TODO: more elegant via SPM!
addpath(pathThis);


%--------------------------------------------------------------------------
% save_dir (directory) - where all data is saved
%--------------------------------------------------------------------------
save_dir         = cfg_files;
save_dir.tag     = 'save_dir';
save_dir.name    = 'save_dir';
save_dir.val     = {{''}};
save_dir.help    = {'Specify a directory where output of modelling and figures shall be saved.'
    'Default: current working directory'};
save_dir.filter  = 'dir';
save_dir.ufilter = '.*';
save_dir.num     = [0 1];

%==========================================================================
%% Sub-structure log_files
%==========================================================================

%--------------------------------------------------------------------------
% vendor
%--------------------------------------------------------------------------
vendor        = cfg_menu;
vendor.tag    = 'vendor';
vendor.name   = 'vendor';
vendor.help   = {' vendor                Name depending on your MR Scanner system'
    '                       ''Philips'''
    '                       ''GE'''
    '                       ''Siemens'''
    '                       ''Siemens_Tics'' - new Siemens physiological'
    '                       logging with time stamps in tics'
    '                       (= steps of 2.5 ms since midnight) and'
    '                       extra acquisition (scan_timing) logfile with'
    '                       time stamps of all volumes and slices'
    ' '
    '                       or'
    '                       ''Custom'''
    ' '
    '  ''Custom'' expects the logfiles (separate files for cardiac and respiratory)'
    '  to be plain text, with one cardiac (or'
    '  respiratory) sample per row;'
    '  If heartbeat (R-wave peak) events are'
    '  recorded as well, they have to be put'
    '  as a 2nd column in the cardiac logfile'
    '  by specifying a 1; 0 in all other rows'
    '  e.g.:'
    '      0.2  0'
    '      0.4  1 <- cardiac pulse event'
    '      0.2  0'
    '      -0.3 0'
    ' '
    ' '
    ' NOTE: the sampling interval has to be specified for these files as'
    ' well (s.b.)'
    };
vendor.labels = {'Philips', 'GE', 'Siemens', 'Siemens_Tics', 'Custom'};
vendor.values = {'Philips', 'GE', 'Siemens', 'Siemens_Tics', 'Custom'};
vendor.val    = {'Philips'};

%--------------------------------------------------------------------------
% cardiac
%--------------------------------------------------------------------------
cardiac         = cfg_files;
cardiac.tag     = 'cardiac';
cardiac.name    = 'log_cardiac';
%cardiac.val     = {{'/Users/kasperla/Documents/code/matlab/smoothing_trunk/tSNR_fMRI_SPM/CheckPhysRETROICOR/PhysIOToolbox/examples/Philips/ECG3T/SCANPHYSLOG.log'}};
cardiac.help    = {'logfile with cardiac, i.e. ECG/PPU (pulse oximetry) data'
    'Select 0 files, if only respiratory data is available'
    'For Philips, same as respiratory logfile.'
    };
cardiac.filter  = 'any';
cardiac.ufilter = '.*';
cardiac.num     = [0 1];

%--------------------------------------------------------------------------
% respiration (filename)
%--------------------------------------------------------------------------
respiration         = cfg_files;
respiration.tag     = 'respiration';
respiration.name    = 'log_respiration';
% respiration.val     = {{'/Users/kasperla/Documents/code/matlab/smoothing_trunk/tSNR_fMRI_SPM/CheckPhysRETROICOR/PhysIOToolbox/examples/Philips/ECG3T/SCANPHYSLOG.log'}};
respiration.help    = {'logfile with respiratory, i.e. breathing belt amplitude data'
    'Select 0 files, if only cardiac data available'
    'For Philips, same as cardiac logfile.'
    };
respiration.filter  = 'any';
respiration.ufilter = '.*';
respiration.num     = [0 1];

%--------------------------------------------------------------------------
% respiration (filename)
%--------------------------------------------------------------------------
log_scan_timing         = cfg_files;
log_scan_timing.tag     = 'scan_timing';
log_scan_timing.name    = 'log_scan_timing';
log_scan_timing.help    = {
    'additional file for relative timing information between logfiles and'
    ' MRI scans.'
    ''
    ' Currently implemented for 2 cases:'
    ' Siemens:      Enter the first or last Dicom volume of your session here,'
    '               The time stamp in the dicom header is on the same time'
    '               axis as the time stamp in the physiological log file'
    ' Siemens_Tics: log-file which holds table conversion for tics axis to' 
    '               time conversion' 
     };
log_scan_timing.filter  = 'any';
log_scan_timing.ufilter = '.*';
log_scan_timing.num     = [0 1];

%--------------------------------------------------------------------------
% sampling_interval
%--------------------------------------------------------------------------
sampling_interval         = cfg_entry;
sampling_interval.tag     = 'sampling_interval';
sampling_interval.name    = 'sampling_interval';
sampling_interval.help    = {
    'sampling interval of phys log files (in seconds)'
    ' If empty, default values are used: 2 ms for Philips, 25 ms for GE and others'
    ' If cardiac and respiratory sampling rate differ, enter them as vector'
    ' [sampling_interval_cardiac, sampling_interval_respiratory]'
    ' '
    ' If cardiac, respiratory and acquisition timing (tics) sampling rate differ,'
    'enter them as a vector:'
    ' [sampling_interval_cardiac, sampling_interval_respiratory sampling_interval_tics_acquisition_timing]'
    ''
    ' Note: If you use a WiFi-Philips device for peripheral monitoring'
    '       (Ingenia system), please change this value to 1/496, '
    '       i.e. a sampling rate of 496 Hz)'
    };
sampling_interval.strtype = 'e';
sampling_interval.num     = [Inf Inf];
sampling_interval.val     = {[]};

%--------------------------------------------------------------------------
% relative_start_acquisition
%--------------------------------------------------------------------------
relative_start_acquisition         = cfg_entry;
relative_start_acquisition.tag     = 'relative_start_acquisition';
relative_start_acquisition.name    = 'relative_start_acquisition';
relative_start_acquisition.help    = {
    'start time (ins seconds) of 1st scan (or dummy)'
    'relative to start of physiological logfile'};
relative_start_acquisition.strtype = 'e';
relative_start_acquisition.num     = [Inf Inf];
relative_start_acquisition.val     = {0};


%--------------------------------------------------------------------------
% align_scan
%--------------------------------------------------------------------------
align_scan        = cfg_menu;
align_scan.tag    = 'align_scan';
align_scan.name   = 'align_scan';
align_scan.help   = { 
    ' Determines which scan shall be aligned to which part of the logfile'
    ' Typically, aligning the last scan to the end of the logfile is'
    ' beneficial, since start of logfile and scans might be shifted due'
    ' to pre-scans'
    ''
    ' NOTE: In all cases, log_files.relative_start_acquisition is'
    '       added to timing after the initial alignmnent to first/last scan'
    ''
    ' ''first''   start of logfile will be aligned to first scan volume'
    ' ''last''    end of logfile will be aligned to last scan volume'
   
    };
align_scan.labels = {'first', 'last'};
align_scan.values = {'first', 'last'};
align_scan.val    = {'last'};


%--------------------------------------------------------------------------
% files
%--------------------------------------------------------------------------
files      = cfg_branch;
files.tag  = 'log_files';
files.name = 'log_files';
files.val  = {vendor cardiac respiration log_scan_timing, ...
    sampling_interval, relative_start_acquisition, align_scan};
files.help = {'Specify log files where peripheral data was stored, and their properties.'};




%==========================================================================
%% Sub-structure sqpar
%==========================================================================



%--------------------------------------------------------------------------
% Nscans
%--------------------------------------------------------------------------
Nscans         = cfg_entry;
Nscans.tag     = 'Nscans';
Nscans.name    = 'Nscans';
Nscans.help    = {
    'Number of scans (volumes) in design matrix.'
    'Put exactly the same number as you have image volumes in your SPM GLM'
    'design specification.'
    };
Nscans.strtype = 'e';
Nscans.num     = [Inf Inf];
%Nscans.val     = {495};

%--------------------------------------------------------------------------
% Ndummies
%--------------------------------------------------------------------------
Ndummies         = cfg_entry;
Ndummies.tag     = 'Ndummies';
Ndummies.name    = 'Ndummies';
Ndummies.help    = {
    'Number of dummies that were acquired (but will not show up in design matrix'
    '(also enter correct number, if dummies are not saved in imaging file)'
    };
Ndummies.strtype = 'e';
Ndummies.num     = [Inf Inf];
%Ndummies.val     = {3};

%--------------------------------------------------------------------------
% TR
%--------------------------------------------------------------------------
TR         = cfg_entry;
TR.tag     = 'TR';
TR.name    = 'TR';
TR.help    = {'Repetition time (in seconds) between consecutive image volumes'};
TR.strtype = 'e';
TR.num     = [Inf Inf];
%TR.val     = {2.5};

%--------------------------------------------------------------------------
% NslicesPerBeat
%--------------------------------------------------------------------------
NslicesPerBeat         = cfg_entry;
NslicesPerBeat.tag     = 'NslicesPerBeat';
NslicesPerBeat.name    = 'NslicesPerBeat';
NslicesPerBeat.help    = {'Only for triggered (gated) sequences: '
    'Number of slices acquired per heartbeat'};
NslicesPerBeat.strtype = 'e';
NslicesPerBeat.num     = [Inf Inf];
NslicesPerBeat.val     = {[]};


%--------------------------------------------------------------------------
% Nslices
%--------------------------------------------------------------------------
Nslices         = cfg_entry;
Nslices.tag     = 'Nslices';
Nslices.name    = 'Nslices';
Nslices.help    = {'Number of slices in one volume'};
Nslices.strtype = 'e';
Nslices.num     = [Inf Inf];
%Nslices.val     = {37};



%--------------------------------------------------------------------------
% onset_slice
%--------------------------------------------------------------------------
onset_slice         = cfg_entry;
onset_slice.tag     = 'onset_slice';
onset_slice.name    = 'onset_slice';
onset_slice.help    = {
    'Slice to which regressors are temporally aligned.'
    'Typically the slice where your most important activation is expected.'};
onset_slice.strtype = 'e';
onset_slice.num     = [Inf Inf];
%onset_slice.val     = {19};

%--------------------------------------------------------------------------
% Nprep
%--------------------------------------------------------------------------
Nprep         = cfg_entry;
Nprep.tag     = 'Nprep';
Nprep.name    = 'Nprep';
Nprep.help    = {
   ' Count of preparation pulses BEFORE 1st dummy scan.' 
    ' Only important, if log_files.scan_align = ''first'', since then'
    ' preparation pulses and dummiy triggers are counted and discarded '
    ' as first scan onset'
     };
Nprep.strtype = 'e';
Nprep.num     = [Inf Inf];
Nprep.val     = {[]};

%--------------------------------------------------------------------------
% time_slice_to_slice
%--------------------------------------------------------------------------
time_slice_to_slice         = cfg_entry;
time_slice_to_slice.tag     = 'time_slice_to_slice';
time_slice_to_slice.name    = 'time_slice_to_slice';
time_slice_to_slice.help    = {
    'Duration between acquisition of two different slices'
    'if empty, set to default value TR/Nslices'
    'differs e.g. if slice timing was minimal and TR was bigger than needed'
    'to acquire Nslices'
    };
time_slice_to_slice.strtype = 'e';
time_slice_to_slice.num     = [Inf Inf];
time_slice_to_slice.val     = {[]};

%--------------------------------------------------------------------------
% sqpar
%--------------------------------------------------------------------------
sqpar      = cfg_branch;
sqpar.tag  = 'sqpar';
sqpar.name = 'sqpar (Sequence timing parameters)';
sqpar.val  = {Nslices NslicesPerBeat TR Ndummies Nscans onset_slice time_slice_to_slice Nprep};
sqpar.help = {'Sequence timing parameters, (number of slices, volumes, dummies, volume TR, slice TR ...)'};




%==========================================================================
%% Sub-structure model
%==========================================================================


%--------------------------------------------------------------------------
% c
%--------------------------------------------------------------------------
c         = cfg_entry;
c.tag     = 'c';
c.name    = 'cardiac';
c.help    = {'Order of Fourier expansion for cardiac phase'
    ' - equals 1/2 number of cardiac regressors, since sine and cosine terms'
    'are computed, i.e. sin(phi), cos(phi), sin(2*phi), cos(2*phi), ..., sin(c*phi), cos(c*phi)'
    };
c.strtype = 'e';
c.num     = [1 1];
c.val     = {3};

%--------------------------------------------------------------------------
% r
%--------------------------------------------------------------------------
r         = cfg_entry;
r.tag     = 'r';
r.name    = 'respiratory';
r.help    = {
    'Order of Fourier expansion for respiratory phase'
    ' - equals 1/2 number of respiratory regressors, since sine and cosine terms'
    'are computed, i.e. sin(phi), cos(phi), sin(2*phi), cos(2*phi), ..., sin(r*phi), cos(r*phi)'
    };
r.strtype = 'e';
r.num     = [1 1];
r.val     = {4};

%--------------------------------------------------------------------------
% cr
%--------------------------------------------------------------------------
cr         = cfg_entry;
cr.tag     = 'cr';
cr.name    = 'cardiac X respiratory';
cr.help    = {
    'Order of Fourier expansion for interaction of cardiac and respiratory phase'
    ' - equals 1/4 number of interaction regressors, since sine and cosine terms'
    'are computed and multiplied, i.e. sin(phi_c)*cos(phi_r), sin(phi_r)*cos(phi_c)'
    };
cr.strtype = 'e';
cr.num     = [1 1];
cr.val     = {1};

%--------------------------------------------------------------------------
% orthog
%--------------------------------------------------------------------------
orthog        = cfg_menu;
orthog.tag    = 'orthogonalise';
orthog.name   = 'orthogonalise';
orthog.help   = {
    'Orthogonalize physiological regressors with respect to each other.'
    'Note: This is only recommended for triggered/gated acquisition sequences.'
    };
orthog.labels = {'none' 'cardiac' 'resp' 'mult' 'all'};
orthog.values = {'none' 'cardiac' 'resp' 'mult' 'all'};
orthog.val    = {'none'};

%--------------------------------------------------------------------------
% order
%--------------------------------------------------------------------------
order      = cfg_branch;
order.tag  = 'order';
order.name = 'order';
order.val  = {c r cr orthog};
order.help = {'...'};

%--------------------------------------------------------------------------
% model_type
%--------------------------------------------------------------------------
model_type        = cfg_menu;
model_type.tag    = 'type';
model_type.name   = 'type';
model_type.help   = {'Physiological Model estimated'};
model_type.labels = {
    'none (only read-in of logfile data into physio.ons_secs)'
    'RETROICOR (RETRO)'
    'Heart Rate Variability (HRV)'
    'Respiratory Volume per Time (RVT)'
    'RETRO+HRV'
    'RETRO+RVT'
    'HRV+RVT'
    'RETRO+HRV+RVT'
    };
model_type.values = {
    'none'
    'RETROICOR'
    'HRV'
    'RVT'
    'RETROICOR_HRV'
    'RETROICOR_RVT'
    'HRV_RVT'
    'RETROICOR_HRV_RVT'
    };
model_type.val    = {'RETROICOR'};

%--------------------------------------------------------------------------
% output_multiple_regressors
%--------------------------------------------------------------------------
output_multiple_regressors         = cfg_entry;
output_multiple_regressors.tag     = 'output_multiple_regressors';
output_multiple_regressors.name    = 'output_multiple_regressors';
output_multiple_regressors.help    = {
    'Output file for physiologica regressors'
    'Choose file name with extension:'
    '.txt for ASCII files with 1 regressor per column'
    '.mat for matlab variable file'
    };
output_multiple_regressors.strtype = 's';
output_multiple_regressors.num     = [1 Inf];
output_multiple_regressors.val     = {'multiple_regressors.txt'};

%--------------------------------------------------------------------------
% input_other_multiple_regressors
%--------------------------------------------------------------------------
input_other_multiple_regressors         = cfg_files;
input_other_multiple_regressors.tag     = 'input_other_multiple_regressors';
input_other_multiple_regressors.name    = 'input_other_multiple_regressors';
input_other_multiple_regressors.val     = {{''}};
input_other_multiple_regressors.help    = {'...'};
input_other_multiple_regressors.filter  = '.*';
input_other_multiple_regressors.ufilter = '.mat$|.txt$';
input_other_multiple_regressors.num     = [0 1];

%--------------------------------------------------------------------------
% model
%--------------------------------------------------------------------------
model      = cfg_branch;
model.tag  = 'model';
model.name = 'model';
model.val  = {model_type, order, input_other_multiple_regressors, ...
    output_multiple_regressors};
model.help = {['Physiological Model to be estimated and Included in GLM ' ... 
    'multiple_regressors.txt']};




% ==========================================================================
%% Sub-structure thresh
%==========================================================================


% ==========================================================================
%% Subsub-structure scan_timing
%==========================================================================


%--------------------------------------------------------------------------
% grad_direction
%--------------------------------------------------------------------------
grad_direction        = cfg_menu;
grad_direction.tag    = 'grad_direction';
grad_direction.name   = 'grad_direction';
grad_direction.help   = {'...'};
grad_direction.labels = {'x' 'y' 'z'};
grad_direction.values = {'x' 'y' 'z'};
grad_direction.val    = {'y'};

%--------------------------------------------------------------------------
% vol_spacing
%--------------------------------------------------------------------------
vol_spacing         = cfg_entry;
vol_spacing.tag     = 'vol_spacing';
vol_spacing.name    = 'vol_spacing';
vol_spacing.help    = {'time (in seconds) between last slice of n-th volume'
    'and 1st slice of n+1-th volume(overrides .vol-threshold)'
    'NOTE: Leave empty if .vol shall be used'};
vol_spacing.strtype = 'e';
vol_spacing.num     = [Inf Inf];
vol_spacing.val     = {[]};

%--------------------------------------------------------------------------
% vol
%--------------------------------------------------------------------------
vol         = cfg_entry;
vol.tag     = 'vol';
vol.name    = 'vol';
vol.help    = {'Gradient Amplitude Threshold for Start of new Volume'};
vol.strtype = 'e';
vol.num     = [Inf Inf];
vol.val     = {[]};

%--------------------------------------------------------------------------
% slice
%--------------------------------------------------------------------------
slice         = cfg_entry;
slice.tag     = 'slice';
slice.name    = 'slice';
slice.help    = {'Gradient Amplitude Threshold for Start of new slice'};
slice.strtype = 'e';
slice.num     = [Inf Inf];
slice.val     = {1800};

%--------------------------------------------------------------------------
% zero
%--------------------------------------------------------------------------
zero         = cfg_entry;
zero.tag     = 'zero';
zero.name    = 'zero';
zero.help    = {'Gradient Amplitude Threshold below which values will be set to 0.'};
zero.strtype = 'e';
zero.num     = [Inf Inf];
zero.val     = {1700};




%--------------------------------------------------------------------------
% scan_timing_method_gradient_log
%--------------------------------------------------------------------------


scan_timing_method_gradient_log = cfg_branch;
scan_timing_method_gradient_log.tag = 'gradient_log';
scan_timing_method_gradient_log.name = 'gradient_log';
scan_timing_method_gradient_log.val  = {
   grad_direction 
   zero 
   slice 
   vol 
   vol_spacing
};
scan_timing_method_gradient_log.help = { ...
    ' Derive scan-timing from logged gradient time courses'
    ' in SCANPHYSLOG-files (Philips only)'};


%--------------------------------------------------------------------------
% scan_timing_method_gradient_log_auto
%--------------------------------------------------------------------------


scan_timing_method_gradient_log_auto = cfg_branch;
scan_timing_method_gradient_log_auto.tag = 'gradient_log_auto';
scan_timing_method_gradient_log_auto.name = 'gradient_log_auto';
scan_timing_method_gradient_log_auto.val  = {};
scan_timing_method_gradient_log_auto.help = { ...
    ' Derive scan-timing from logged gradient time courses'
    ' in SCANPHYSLOG-files automatically (Philips only), '
    ' using prior information on TR and number of slices, '
    'i.e. without manual threshold settings.'
};


%--------------------------------------------------------------------------
% scan_timing_method_nominal
%--------------------------------------------------------------------------

scan_timing_method_nominal = cfg_branch;
scan_timing_method_nominal.tag = 'nominal';
scan_timing_method_nominal.name = 'nominal';
scan_timing_method_nominal.val  = {};
scan_timing_method_nominal.help = { ...
    ' Derive scan-timing for sqpar (nominal scan timing parameters)'};


%--------------------------------------------------------------------------
% scan_timing_method_scan_timing_log
%--------------------------------------------------------------------------

scan_timing_method_scan_timing_log = cfg_branch;
scan_timing_method_scan_timing_log.tag = 'scan_timing_log';
scan_timing_method_scan_timing_log.name = 'scan_timing_log';
scan_timing_method_scan_timing_log.val  = {};
scan_timing_method_scan_timing_log.help = { ...
    ' Derive scan-timing from individual scan timing logfile with time '
    ' stamps ("tics") for each slice and volume (e.g. Siemens_Cologne)'};

 
 

%--------------------------------------------------------------------------
% scan_timing
%--------------------------------------------------------------------------
scan_timing      = cfg_choice;
scan_timing.tag  = 'scan_timing';
scan_timing.name = 'Scan/Physlog Time Synchronization';
scan_timing.values  = {scan_timing_method_nominal, ...
    scan_timing_method_gradient_log, ...
    scan_timing_method_gradient_log_auto, ...
    scan_timing_method_scan_timing_log};
scan_timing.val = {scan_timing_method_nominal};
scan_timing.help = {'Determines scan timing from nominal scan parameters or logged gradient time courses'
    ''
' Available methods to determine slice onset times'
' ''nominal''         - to derive slice acquisition timing from sqpar directly'
' ''gradient_log''    - derive from logged gradient time courses'
'                                in SCANPHYSLOG-files (Philips only)'
' ''gradient_log_auto'' - as gradient_log, but thresholds are determined'
'                         automatically from TR and number of slices expected' 
' ''scan_timing_log'' - individual scan timing logfile with time stamps ("tics") for each slice and volume (e.g. Siemens_Cologne)'
};




% ==========================================================================
%% Subsub-structure cardiac
%==========================================================================


%--------------------------------------------------------------------------
% modality
%--------------------------------------------------------------------------
modality        = cfg_menu;
modality.tag    = 'modality';
modality.name   = 'modality';
modality.help   = {'Shall ECG or PPU data be read from logfiles?'};
modality.labels = {'ECG', 'OXY/PPU', 'ECG_WiFi', 'PPU_WiFi'};
modality.values = {'ECG', 'PPU', 'ECG_WiFi', 'PPU_Wifi'};
modality.val    = {'ECG'};



%--------------------------------------------------------------------------
% initial_cpulse_select_file
%--------------------------------------------------------------------------
initial_cpulse_select_file         = cfg_entry;
initial_cpulse_select_file.tag     = 'file';
initial_cpulse_select_file.name    = 'file';
initial_cpulse_select_file.help    = {'...'};
initial_cpulse_select_file.strtype = 's';
initial_cpulse_select_file.num     = [0 Inf];
initial_cpulse_select_file.val     = {'initial_cpulse_kRpeakfile.mat'};



%--------------------------------------------------------------------------
% min
%--------------------------------------------------------------------------
min       = cfg_entry;
min.tag     = 'min';
min.name    = 'min';
min.help    = {'Minimum correlation value considered a peak (for auto, manual, load-methods).'};
min.strtype = 'e';
min.num     = [Inf Inf];
min.val     = {0.4};


%--------------------------------------------------------------------------
% initial_cpulse_select_method_auto_template
%--------------------------------------------------------------------------

initial_cpulse_select_method_auto_template = cfg_branch;
initial_cpulse_select_method_auto_template.tag = 'auto_template';
initial_cpulse_select_method_auto_template.name = 'auto_template';
initial_cpulse_select_method_auto_template.val  = {
    min 
    initial_cpulse_select_file    
};
initial_cpulse_select_method_auto_template.help = { ...
    ' Auto generation of representative QRS-wave; detection via'
    '             maximising auto-correlation with it (default)'};




%--------------------------------------------------------------------------
% initial_cpulse_select_method_auto_matched
%--------------------------------------------------------------------------

initial_cpulse_select_method_auto_matched = cfg_branch;
initial_cpulse_select_method_auto_matched.tag = 'auto_matched';
initial_cpulse_select_method_auto_matched.name = 'auto_matched';
initial_cpulse_select_method_auto_matched.val  = {
    min 
    initial_cpulse_select_file    
};
initial_cpulse_select_method_auto_matched.help = { ...
    'Auto generation of template QRS wave, '
    '             matched-filter/autocorrelation detection of heartbeats'
    };




%--------------------------------------------------------------------------
% initial_cpulse_select_method_manual_template
%--------------------------------------------------------------------------

initial_cpulse_select_method_manual_template = cfg_branch;
initial_cpulse_select_method_manual_template.tag = 'manual_template';
initial_cpulse_select_method_manual_template.name = 'manual_template';
initial_cpulse_select_method_manual_template.val  = {
    min 
    initial_cpulse_select_file    
};
initial_cpulse_select_method_manual_template.help = { ...
    'Manually select QRS-wave for autocorrelation detection'
    };


%--------------------------------------------------------------------------
% initial_cpulse_select_method_load_template
%--------------------------------------------------------------------------

initial_cpulse_select_method_load_template = cfg_branch;
initial_cpulse_select_method_load_template.tag = 'load_template';
initial_cpulse_select_method_load_template.name = 'load_template';
initial_cpulse_select_method_load_template.val  = {
    min 
    initial_cpulse_select_file    
};
initial_cpulse_select_method_load_template.help = { ...
    'Load template from previous manual/auto run to perform autocorrelation detection of hearbeats'
    };



%--------------------------------------------------------------------------
% initial_cpulse_select_method_load_from_logfile
%--------------------------------------------------------------------------

initial_cpulse_select_method_load_from_logfile = cfg_branch;
initial_cpulse_select_method_load_from_logfile.tag = 'load_from_logfile';
initial_cpulse_select_method_load_from_logfile.name = 'load_from_logfile';
initial_cpulse_select_method_load_from_logfile.val  = {};
initial_cpulse_select_method_load_from_logfile.help = { ...
    'Load heartbeat data from Phys-logfile, detected R-peaks of scanner'};


%--------------------------------------------------------------------------
% initial_cpulse_select
%--------------------------------------------------------------------------
initial_cpulse_select      = cfg_choice;
initial_cpulse_select.tag  = 'initial_cpulse_select';
initial_cpulse_select.name = 'Initial Detection of Heartbeats';
initial_cpulse_select.val  = {initial_cpulse_select_method_load_from_logfile};
initial_cpulse_select.values  = {
    initial_cpulse_select_method_auto_matched, ...
    initial_cpulse_select_method_auto_template, ...
    initial_cpulse_select_method_load_from_logfile, ...
    initial_cpulse_select_method_manual_template, ...
    initial_cpulse_select_method_load_template, ...
    };



initial_cpulse_select.help = {
    'The initial cardiac pulse selection structure: Determines how the'
    'majority of cardiac pulses is detected in a first pass.'
    ' ''auto_matched''     - auto generation of template QRS wave, '
    '             matched-filter/autocorrelation detection of heartbeats'
    ' ''auto_template''    - auto generation of representative QRS-wave; detection via'
    '             maximising auto-correlation with it (default)'
    ' ''load_from_logfile'' - from phys logfile, detected R-peaks of scanner'
    ' ''manual_template''  - via manually selected QRS-wave for autocorrelations'
    ' ''load_template''    - from previous manual/auto run'
    };


%--------------------------------------------------------------------------
% posthoc_cpulse_select_file
%--------------------------------------------------------------------------
posthoc_cpulse_select_file         = cfg_entry;
posthoc_cpulse_select_file.tag     = 'file';
posthoc_cpulse_select_file.name    = 'file';
posthoc_cpulse_select_file.help    = {'...'};
posthoc_cpulse_select_file.strtype = 's';
posthoc_cpulse_select_file.num     = [0 Inf];
posthoc_cpulse_select_file.val     = {'posthoc_cpulse.mat'};



%--------------------------------------------------------------------------
% posthoc_cpulse_select_percentile
%--------------------------------------------------------------------------
posthoc_cpulse_select_percentile       = cfg_entry;
posthoc_cpulse_select_percentile.tag     = 'percentile';
posthoc_cpulse_select_percentile.name    = 'percentile';
posthoc_cpulse_select_percentile.help    = {
    'percentile of beat-2-beat interval histogram that constitutes the'
    'average heart beat duration in the session'};
posthoc_cpulse_select_percentile.strtype = 'e';
posthoc_cpulse_select_percentile.num     = [Inf Inf];
posthoc_cpulse_select_percentile.val     = {80};

%--------------------------------------------------------------------------
% posthoc_cpulse_select_upper_thresh
%--------------------------------------------------------------------------
posthoc_cpulse_select_upper_thresh       = cfg_entry;
posthoc_cpulse_select_upper_thresh.tag     = 'upper_thresh';
posthoc_cpulse_select_upper_thresh.name    = 'upper_thresh';
posthoc_cpulse_select_upper_thresh.help    = {
    'minimum exceedance (in %) from average heartbeat duration '
    'to be classified as missing heartbeat'};
posthoc_cpulse_select_upper_thresh.strtype = 'e';
posthoc_cpulse_select_upper_thresh.num     = [Inf Inf];
posthoc_cpulse_select_upper_thresh.val     = {60};

%--------------------------------------------------------------------------
% posthoc_cpulse_select_lower_thresh
%--------------------------------------------------------------------------
posthoc_cpulse_select_lower_thresh       = cfg_entry;
posthoc_cpulse_select_lower_thresh.tag     = 'lower_thresh';
posthoc_cpulse_select_lower_thresh.name    = 'lower_thresh';
posthoc_cpulse_select_lower_thresh.help    = {
    'minimum reduction (in %) from average heartbeat duration'
    'to be classified an abundant heartbeat'};
posthoc_cpulse_select_lower_thresh.strtype = 'e';
posthoc_cpulse_select_lower_thresh.num     = [Inf Inf];
posthoc_cpulse_select_lower_thresh.val     = {60};


%--------------------------------------------------------------------------
% posthoc_cpulse_select_method_off
%--------------------------------------------------------------------------

posthoc_cpulse_select_method_off         = cfg_branch;
posthoc_cpulse_select_method_off.tag  = 'off';
posthoc_cpulse_select_method_off.name = 'Off';
posthoc_cpulse_select_method_off.val  = {};
posthoc_cpulse_select_method_off.help = {'No manual post-hoc pulse selection'};



%--------------------------------------------------------------------------
% posthoc_cpulse_select_method_manual
%--------------------------------------------------------------------------

posthoc_cpulse_select_method_manual      = cfg_branch;
posthoc_cpulse_select_method_manual.tag  = 'manual';
posthoc_cpulse_select_method_manual.name = 'Manual';
posthoc_cpulse_select_method_manual.help = {'Manual post-hoc cardiac pulse selection by clicking'};

posthoc_cpulse_select_method_manual.val = {...
    posthoc_cpulse_select_file ...
    posthoc_cpulse_select_percentile ...
    posthoc_cpulse_select_upper_thresh ...
    posthoc_cpulse_select_lower_thresh};



%--------------------------------------------------------------------------
% posthoc_cpulse_select_method_load
%--------------------------------------------------------------------------

posthoc_cpulse_select_method_load      = cfg_branch;
posthoc_cpulse_select_method_load.tag  = 'load';
posthoc_cpulse_select_method_load.name = 'Load';
posthoc_cpulse_select_method_load.help = {'Loads manually selected cardiac pulses from file'};
posthoc_cpulse_select_method_load.val = {
    posthoc_cpulse_select_file};


%--------------------------------------------------------------------------
% posthoc_cpulse_select
%--------------------------------------------------------------------------

posthoc_cpulse_select      = cfg_choice;
posthoc_cpulse_select.tag  = 'posthoc_cpulse_select';
posthoc_cpulse_select.name = 'Post-hoc Selection of Cardiac Pulses';
posthoc_cpulse_select.val  = {posthoc_cpulse_select_method_off};
posthoc_cpulse_select.values = {posthoc_cpulse_select_method_off, ...
    posthoc_cpulse_select_method_manual, ...
    posthoc_cpulse_select_method_load};
    

posthoc_cpulse_select.help = {
    'The posthoc cardiac pulse selection structure: If only few (<20)'
    'cardiac pulses are missing in a session due to bad signal quality, a'
    'manual selection after visual inspection is possible using the'
    'following parameters. The results are saved for reproducibility.'
    ''
    'Refers to physio.thresh.cardiac.posthoc_cpulse_select.method in physio-structure'
    };



%--------------------------------------------------------------------------
% cardiac
%--------------------------------------------------------------------------
cardiac      = cfg_branch;
cardiac.tag  = 'cardiac';
cardiac.name = 'cardiac';
cardiac.val  = {modality initial_cpulse_select posthoc_cpulse_select};
cardiac.help = {'...'};


%--------------------------------------------------------------------------
% thresh
%--------------------------------------------------------------------------
thresh      = cfg_branch;
thresh.tag  = 'thresh';
thresh.name = 'thresh (Thresholding parameters for de-noising and timing)';
thresh.val  = {scan_timing cardiac};
thresh.help = {'Thresholding parameters for de-noising of raw peripheral data'
    'and determination of sequence timing from logged MR gradient time courses'};



%==========================================================================
%% Sub-structure verbose
%==========================================================================

%--------------------------------------------------------------------------
% level
%--------------------------------------------------------------------------
level         = cfg_entry;
level.tag     = 'level';
level.name    = 'level';
level.help    = {'...'};
level.strtype = 'e';
level.num     = [Inf Inf];
level.val     = {2};

%--------------------------------------------------------------------------
% fig_output_file
%--------------------------------------------------------------------------
fig_output_file         = cfg_entry;
fig_output_file.tag     = 'fig_output_file';
fig_output_file.name    = 'fig_output_file';
fig_output_file.help    = {'file name where figures are saved to;'
    'supported figure formats(via filename-suffix): jpg, png, fig, ps'
    'leave empty to not save output figures'};
fig_output_file.strtype = 's';
fig_output_file.num     = [0 Inf];
fig_output_file.val     = {''};




%--------------------------------------------------------------------------
% use_tabs
%--------------------------------------------------------------------------
use_tabs        = cfg_menu;
use_tabs.tag    = 'use_tabs';
use_tabs.name   = 'use_tabs';
use_tabs.help   = {'use spm_tabs for plotting'};
use_tabs.labels = {'true' 'false'};
use_tabs.values = {true, false};
use_tabs.val    = {false};


%--------------------------------------------------------------------------
% verbose
%--------------------------------------------------------------------------
verbose        = cfg_branch;
verbose.tag    = 'verbose';
verbose.name   = 'verbose';
verbose.help   = {
    ' determines how many figures shall be generated to follow the workflow'
    ' of the toolbox and whether the graphical output shall be saved (to a'
    ' PostScript-file)'
    ' 0 = no graphical output;'
    ' 1 = (default) main plots : Fig 1: gradient scan timing (if selected) ;'
    '                            Fig 2: heart beat/breathing statistics & outlier;'
    '                            Fig 3: final multiple_regressors matrix'
    ' 2 = debugging plots        for setting up new study or if Fig 2 had'
    '                            outliers'
    '                            Fig 1: raw phys logfile data'
    '                            Fig 2: gradient scan timing (if selected)'
    '                            Fig 3: cutout interval of logfile for'
    '                            regressor creation (including scan timing'
    '                            and raw phys data)'
    '                            Fig 4: heart beat/breathing statistics & outlier;'
    '                            Fig 5: time course of all sampled RETROICOR'
    '                                   regressors'
    '                            Fig 6: final multiple_regressors matrix'
    ''
    ' 3 = all plots'
    '                            Fig 1: raw phys logfile data'
    '                            Fig 2: gradient scan timing (if selected)'
    '                            Fig 3: Slice assignment to volumes'
    '                            Fig 4: cutout interval of logfile for'
    '                            regressor creation (including scan timing'
    '                            and raw phys data)'
    '                            Fig 5: heart beat/breathing statistics & outlier;'
    '                            Fig 6: cardiac phase data of all slices'
    '                            Fig 7: respiratory phase data and'
    '                                   histogram transfer function'
    '                            Fig 8: time course of all sampled RETROICOR'
    '                                   regressors'
    '                            Fig 9: final multiple_regressors matrix'
    
    };
verbose.val    = {level fig_output_file use_tabs};




%==========================================================================
%% Structure physio Assemblance
%==========================================================================


%--------------------------------------------------------------------------
% physio
%--------------------------------------------------------------------------
physio      = cfg_exbranch;
physio.tag  = 'physio';
physio.name = 'TAPAS PhysIO Toolbox';
physio.val  = {save_dir files sqpar thresh model verbose};
physio.help = {'...'};
physio.prog = @run_physio;
physio.vout = @vout_physio;


%==========================================================================
% function out = run_physio(job)
%==========================================================================
function out = run_physio(job)

%% Rename job fields to the ones actually used in physio-structure

physio = tapas_physio_job2physio(job);

[physio_out, R] = tapas_physio_main_create_regressors(physio);

out.physnoisereg = cellstr(physio_out.model.output_multiple_regressors);
out.R = R;


%==========================================================================
% function dep = vout_physio(job)
%==========================================================================
function dep = vout_physio(job)
dep(1)            = cfg_dep;
dep(1).sname      = 'physiological noise regressors file';
dep(1).src_output = substruct('.','physnoisereg');
dep(1).tgt_spec   = cfg_findspec({{'filter','mat','strtype','e'}});
