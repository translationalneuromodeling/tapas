function physio = tapas_physio_new(default_scheme, physio_in)
% creates complete PhysIO structure fed into tapas_physio_main_create_regressors
%
%    physio = tapas_physio_new(default_scheme, physio_in)
%
% IN
%   default_scheme  - if set, default values for structure entries are set
%                       according to the application
%                       different templates are predefined, e.g.
%                       'empty'   - all strings are set to '', all
%                                     numbers to [] (default)
%                       'Philips':  good initial values for scans acquired
%                                   on a Philips 3T system
%                               model: RETROICOR;
%                               vendor: Philips;
%                               heartbeat detection: ECG, load_from_logfile
%                                                    Philips detected peaks
%                                                    no posthoc-detection
%                               scan_timing:         gradient_log
%                                                    uses gradient data
%                                                    from SCANPHYSLOG-file
%
%                       'RETROICOR' order of RETROICOR expansion taken from
%                       Harvey2008, JRMI28(6), p1337ff.
%                       'scan_timing_from_start'
%                       'manual_peak_select'
%   physio_in       - used as input, only fields related to default_scheme
%                     are overwritten, the others are kept as in physio_in
%
% OUT
%   physio          - the complete physio structure, which can be unsed in
%                     tapas_physio_main_create_regressors
%
% NOTE
%   All parameters used in the physIO toolbox are defined AND DOCUMENTED in
%   this file. Just scroll down and read through the comments!
%
% EXAMPLE
%   physio = tapas_physio_new('default')
%       OR = tapas_physio_new():
%
%   physio = tapas_physio_new('empty')
%   physio = tapas_physio_new('RETROICOR');
%   physio = tapas_physio_new('manual_peak_select', physio);
%
%   See also tapas_physio_main_create_regressors
%
% Author: Lars Kasper
% Created: 2013-04-23
% Copyright (C) 2013 TNU, Institute for Biomedical Engineering, University of Zurich and ETH Zurich.
%
% This file is part of the TAPAS PhysIO Toolbox, which is released under the terms of the GNU General Public
% Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
% (either version 3 or, at your option, any later version). For further details, see the file
% COPYING or <http://www.gnu.org/licenses/>.
%
% $Id: tapas_physio_new.m 668 2015-02-01 12:22:26Z kasperla $

% if not specified differently, create everything empty
if ~nargin
    default_scheme = 'empty';
end




if nargin >= 2
    save_dir = physio_in.save_dir;
    log_files   = physio_in.log_files;
    thresh  = physio_in.thresh;
    sqpar   = physio_in.sqpar;
    model   = physio_in.model;
    verbose = physio_in.verbose;
    ons_secs = physio_in.ons_secs;
else
    
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% save_dir
    % directory where output model and figure-files are saved to
    % leave empty to use current directory
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    save_dir = '';
    
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% log_files
    % Structure containing general physiological log-file information
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    log_files              = [];
    
    % vendor                Name depending on your MR Scanner system
    %                       'Philips' (default)
    %                       'GE',
    %                       'Siemens'
    %                       'Siemens_Tics' - new Siemens physiological
    %                       logging with time stamps in tics
    %                       (= steps of 2.5 ms since midnight) and
    %                       extra acquisition (scan_timing) logfile with
    %                       time stamps of all volumes and slices
    %
    %                       or
    %                       'Custom'
    %
    %  'Custom' expects the logfiles (separate files for cardiac and respiratory)
    %  to be plain text, with one cardiac (or
    %  respiratory) sample per row;
    %  If heartbeat (R-wave peak) events are
    %  recorded as well, they have to be put
    %  as a 2nd column in the cardiac logfile
    %  by specifying a 1; 0 in all other rows
    %  e.g.:
    %      0.2  0
    %      0.4  1 <- cardiac pulse event
    %      0.2  0
    %      -0.3 0
    %
    %
    % NOTE: the sampling interval has to be specified for these files as
    % well (s.b.)
    
    log_files.vendor       = 'Philips';
    
    log_files.cardiac      = ''; % 'SCANPHYSLOG.log'; logfile with cardiac data
    
    % additional file for relative timing information between logfiles and
    % MRI scans.
    % Currently implemented for 2 cases
    % Siemens:      Enter the first or last Dicom volume of your session here,
    %               The time stamp in the dicom header is on the same time
    %               axis as the time stamp in the physiological log file
    % Siemens_Tics: log-file which holds table conversion for tics axis to 
    %               time conversion 
    log_files.scan_timing  = ''; 
    
    % Logfile with respiratory data, e.g. 'SCANPHYSLOG.log';
    % (same as .cardiac for Philips)
    log_files.respiration  = '';
    
    % Sampling interval in seconds (i.e. time between two rows in logfile
    % if empty, default value will be set: 2e-3 for Philips, variable for GE, e.g. 40e-3
    %         1 entry: sampling interval (seconds)
    %         for both cardiac + respiratory log file
    %         2 entries: 1st entry sampling interval (seconds)
    %         for cardiac logfile, 2nd entry for respiratory
    %         logfile
    log_files.sampling_interval = [];
    
    % Time (in seconds) when the 1st scan (or, if existing, dummy) started,
    % relative to the start of the logfile recording;
    % e.g.  0 if simultaneous start
    %       10, if 1st scan starts 10
    %       seconds AFTER physiological
    %       recording
    %       -20, if first scan started 20
    %       seconds BEFORE phys recording
    % NOTE: For Philips SCANPHYSLOG, this parameter is ignored, if
    %       thresh.scan_timing is set
    log_files.relative_start_acquisition = 0;
    
    % Determines which scan shall be aligned to which part of the logfile
    % Typically, aligning the last scan to the end of the logfile is
    % beneficial, since start of logfile and scans might be shifted due
    % to pre-scans
    %
    % NOTE: In all cases, log_files.relative_start_acquisition is
    %       added to timing after the initial alignmnent to first/last scan
    %
    % 'first'   start of logfile will be aligned to first scan volume
    % 'last'    end of logfile will be aligned to last scan volume
    log_files.align_scan       = 'last'; 

    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% sqpar - Parameters for sequence timing, i.e. slice and volume
    %  acquisition starts, TR, number of scans etc.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    sqpar = [];
    
    sqpar.Nslices           = [];   % number of slices per volume in fMRI scan
    sqpar.NslicesPerBeat    = [];   % usually equals Nslices, unless you trigger with the heart beat
    sqpar.TR                = [];   % volume repetition time in seconds
    sqpar.Ndummies          = [];   % number of dummy volumes
    
    % number of full volumes saved (volumes in nifti file,
    % usually rows in your design matrix)
    sqpar.Nscans            = [];
        
    % Count of preparation pulses
    % BEFORE 1st dummy scan. 
    % Only important, if log_files.scan_align = 'first', since then
    % preparation pulses and dummiy triggers are counted and discarded 
    % as first scan onset
    sqpar.Nprep             = [];
    
    % time between the acquisition of 2 subsequent
    % slices; typically TR/Nslices or minTR/Nslices,
    % if minimal temporal slice spacing was chosen
    % NOTE: only necessary, if thresh.grad_direction
    % is empty and nominal scan timing is used
    sqpar.time_slice_to_slice  = [];
    
    % slice whose scan onset determines the adjustment of the
    % regressor timing to a particular slice for the whole volume
    % volumes from beginning of run, i.e. logfile,
    % includes counting of preparation gradients
    sqpar.onset_slice       = [];
    
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% model
    % Determines the physiological noise model derived from preprocessed physiological data
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    model = [];
    
    % 'none'        no physiological model is computed; only the read-out
    %               logfile data is read out and saved in physio.ons_secs
    % 'RETROICOR'   as in Glover el al, MRM 44, 2000l
    %               order of expansion:  See Harvey et al, JMRI 28, 2008
    % 'HRV'         heart rate variability, as in Chang et al, 2009
    % 'RVT'         respiratory volume time, as in Birn et al., 2006
    %
    % The above can be combined e.g. 'RETROICOR_HRV', 'RETROICOR_RVT',
    % 'RETROICOR_HRV_RVT, 'HRV_RVT'
    model.type = '';
    
    % other nuisance regressors to be included in design matrix
    % either txt-file or mat-file with variable R
    model.input_other_multiple_regressors = '';
    
    % output file for usage in SPM multiple_regressors GLM-specification
    % either txt-file or mat-file with variable R
    model.output_multiple_regressors = '';
    
    % natural number, order of cardiac phase Fourier expansion
    model.order.c = [];
    
    % natural number, order of respiratory phase Fourier expansion
    model.order.r = [];
    
    % natural number, order of cardiac-respiratory-phase-interaction Fourier expansion
    model.order.cr = [];
    
    model.order.orthogonalise = 'none';         % string indicating which regressors shall be orthogonalised;
    % mainly needed, if acquisition was triggered to heartbeat (set to 'cardiac') OR
    % if session mean shall be evaluated (e.g. SFNR-studies, set to 'all')
    % 'n' or 'none'     - no orthogonalisation is performed
    % Possible Values (default: 'none'
    %   'c' or 'cardiac'  - only cardiac regressors are orthogonalised
    %   'r' or 'resp'     - only respiration regressors are orthogonalised
    %   'mult'            - only multiplicative regressors are orthogonalised
    %   'all'             - all physiological regressors are orthogonalised to each other
    model.R = [];                               % output design matrix of confound regressors, saved in 'multiple_regressors.mat'
    
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% thresh
    % Determines thresholds used in preprocessing physiological logfiles,
    % either their timing (thresh.scan_timing) or the peripheral measures
    % itself (thresh.cardiac, thresh.respiration)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    thresh = [];
    
    % Method to determine slice acquisition onset times
    % 'scan_timing_log'     individual scan timing logfile with time stamps
    %                       ("tics") for each slice and volume
    %                       (e.g. Siemens_Cologne)
    % 'nominal'             derive slice acquisition timing from sqpar
    %                       directly
    % 'gradient'            derive from logged gradient time courses
    %   or 'gradient_log'   in SCANPHYSLOG-files (Philips only)
    % 'gradient_auto'       derive from logged gradient time courses
    %   or 'gradient_log_auto'   in SCANPHYSLOG-files automatically, i.e.
    %                       without defining thresholds (Philips only)
    thresh.scan_timing.method = 'gradient_log';
    thresh.scan_timing.grad_direction = ''; % 'x', 'y', or 'z';
    
    % if set, sequence timing is calculated
    % from logged gradient timecourse along
    % this coordinate axis;
    
    thresh.scan_timing.zero     = [];   % gradient values below this value are set to zero;
    
    % should be those which are unrelated to slice acquisition start
    
    % minimum gradient amplitude to be exceeded when a slice scan starts
    thresh.scan_timing.slice    = [];
    
    % minimum gradient amplitude to be exceeded when a new volume starts;
    % leave [], if volume events shall be determined as
    % every Nslices-th scan event or via vol_spacing
    thresh.scan_timing.vol      = [];
    
    
    % duration (in seconds) from last slice acq to
    % first slice of next volume;
    % leave [], if .vol-threshold shall be used
    thresh.scan_timing.vol_spacing          = [];
    
    thresh.cardiac = [];
    thresh.cardiac.modality = 'ecg_wifi'; % 'ECG','ECG_raw', or 'OXY'/'PPU' (for pulse oximetry), 'OXY_OLD', [deprecated]
    
    % The initial cardiac pulse selection structure: Determines how the
    % majority of cardiac pulses is detected
    % 'auto'    - auto generation of representative QRS-wave; detection via
    %             maximising auto-correlation with it
    % 'load_from_logfile' - from phys logfile, detected R-peaks of scanner
    % 'manual'  - via manually selected QRS-wave for autocoreelations
    % 'load'    - from previous manual/auto run
    thresh.cardiac.initial_cpulse_select.method = 'load_from_logfile';
    
    % file containing reference ECG-peak (variable kRpeak)
    % used for method 'manual' or 'load' [default: not set]
    % if method == 'manual', this file is saved after picking the QRS-wave
    % such that results are reproducible
    thresh.cardiac.initial_cpulse_select.file = '';
    
    % threshold for correlation with QRS-wave to find cardiac pulses
    thresh.cardiac.initial_cpulse_select.min = [];
    
    % variable saving an example cardiac QRS-wave to correlate with
    % ECG time series
    thresh.cardiac.initial_cpulse_select.kRpeak = [];
    
    % The posthoc cardiac pulse selection structure: If only few (<20)
    % cardiac pulses are missing in a session due to bad signal quality, a
    % manual selection after visual inspection is possible using the
    % following parameters. The results are saved for reproducibility
    %
    % 'off'     - no manual selection of peaks
    % 'manual'  - pick and save additional peaks manually
    % 'load'    - load previously selected cardiac pulses
    thresh.cardiac.posthoc_cpulse_select.method = 'off';
    
    % filename where cardiac pulses are saved after manual picking
    thresh.cardiac.posthoc_cpulse_select.file = '';
    
    % Suspicious positions of missing or too many cardiac pulses are
    % pre-selected by detecting outliers in histogram of
    % heart-beat-2-beat-intervals
    thresh.cardiac.posthoc_cpulse_select.percentile = 80; % percentile of beat-2-beat interval histogram that constitutes the "average heart beat duration" in the session
    thresh.cardiac.posthoc_cpulse_select.upper_thresh = 60; % minimum exceedance (in %) from average heartbeat duration to be classified as missing heartbeat
    thresh.cardiac.posthoc_cpulse_select.lower_thresh = 60; % minimum reduction (in %) from average heartbeat duration to be classified an abundant heartbeat
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% verbose
    % determines how many figures shall be generated to follow the workflow
    % of the toolbox and whether the graphical output shall be saved (to a
    % PostScript-file)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    verbose = [];
    
    % 0 = no graphical output;
    % 1 = (default) main plots : Fig 1: gradient scan timing (if selected) ;
    %                            Fig 2: heart beat/breathing statistics & outlier;
    %                            Fig 3: final multiple_regressors matrix
    % 2 = debugging plots        for setting up new study or if Fig 2 had
    %                            outliers
    %                            Fig 1: raw phys logfile data
    %                            Fig 2: gradient scan timing (if selected)
    %                            Fig 3: cutout interval of logfile for
    %                            regressor creation (including scan timing
    %                            and raw phys data)
    %                            Fig 4: heart beat/breathing statistics & outlier;
    %                            Fig 5: time course of all sampled RETROICOR
    %                                   regressors
    %                            Fig 6: final multiple_regressors matrix
    %
    % 3 = all plots
    %                            Fig 1: raw phys logfile data
    %                            Fig 2: gradient scan timing (if selected)
    %                            Fig 3: Slice assignment to volumes
    %                            Fig 4: cutout interval of logfile for
    %                            regressor creation (including scan timing
    %                            and raw phys data)
    %                            Fig 5: heart beat/breathing statistics & outlier;
    %                            Fig 6: cardiac phase data of all slices
    %                            Fig 7: respiratory phase data and
    %                                   histogram transfer function
    %                            Fig 8: time course of all sampled RETROICOR
    %                                   regressors
    %                            Fig 9: final multiple_regressors matrix
    verbose.level = 1;
    verbose.fig_handles = [];     % [nFigs,1] vector; collecting of all generated figure handles during a run of tapas_physio_main_create_regressors
    verbose.fig_output_file = ''; % file name (including extension) where to print all physIO output figures to,
    verbose.use_tabs = false;    % if true, plots are performed in tabs of SPM graphics window
    %                                   TODO: implement via
    %                               [handles] = spm_uitab(hparent,labels,callbacks,...
    %                                           tag,active,height,tab_height)
    %
    % e.g. 'PhysIO_output.ps' or 'PhysIO_output.jpg'
    % The specified extension determines how the
    % figures will be saved
    %     .ps - all figures are saved to the
    %     same, multiple-page postscript-file
    %     .fig, .tiff,  .jpg
    %         - one file is created for each
    %         figure, appended by its figure
    %         index, e.g. 'PhysIO_output_fig01.jpg'
    
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% ons_secs
    % output structure holding all time-dependent variables, i.e. onsets, specified in seconds
    % all elements but .raw are cropped to the acquisition window of
    % interest
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ons_secs                     = [];
    
    % read-in data
    ons_secs.t              	 = [];  % time vector corresponding to c and r
    ons_secs.c              	 = [];  % raw cardiac waveform (ECG or PPU)
    ons_secs.r              	 = [];  % raw respiration amplitude time course
    ons_secs.c_scaling           = 1;   % stores scaling factor for cardiac data
                                        % after normalization
    ons_secs.r_scaling           = 1;   % stores scaling factor for respiratory data
                                        % after normalization
                                        
    % processed elements cardiac pulse detecion and phase estimations
    ons_secs.cpulse         	 = [];  % onset times of cardiac pulse events (e.g. R-peaks)
    ons_secs.c_sample_phase      = [];  % phase in heart-cycle when each slice of each volume was acquired
    ons_secs.fr                  = [];  % filtered respiration amplitude time series
    ons_secs.hr                  = [];  % [nScans,1] estimated heart rate at each scan
    ons_secs.r_sample_phase      = [];  % phase in respiratory cycle when each slice of each volume was acquired
    
    % scan timing parameters
    ons_secs.svolpulse      	 = [];  % [Nscans x 1] onset times of volume scan events
    ons_secs.spulse         	 = [];  % [Nscans*Nslices x 1] onset times of slice (incl. volume) scan events
    ons_secs.spulse_per_vol 	 = [];  % cell(Nscans,1), as spulse, holding slice scan events sorted by volume
    
    % uncropped parameters
    ons_secs.raw            	 = [];  % raw read-in version of the whole structure, before any cropping
end

switch default_scheme
    case 'RETROICOR'
        model.type = 'RETROICOR';
        model.order = struct('c',3,'r',4,'cr',1, 'orthogonalise', 'none');
    case 'redetect_peaks_from_logfile'
        thresh.cardiac.initial_cpulse_select.method = 'manual'; % 'load_from_logfile', 'manual', 'load'
        thresh.cardiac.initial_cpulse_select.file = 'kRpeak.mat';
        thresh.cardiac.initial_cpulse_select.min = 1;
        thresh.cardiac.initial_cpulse_select.kRpeak = [];
    case 'manual_peak_select'
        thresh.cardiac.posthoc_cpulse_select.method = 'manual'; % 'off', 'manual' or 'load',
        thresh.cardiac.posthoc_cpulse_select.file = 'posthoc_cpulse.mat';
        thresh.cardiac.posthoc_cpulse_select.percentile = 80;
        thresh.cardiac.posthoc_cpulse_select.upperThresh = 60;
        thresh.cardiac.posthoc_cpulse_select.lowerThresh = 30;
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Assemble output
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

physio.save_dir = save_dir;
physio.log_files   = log_files;
physio.sqpar   = sqpar;
physio.model   = model;
physio.thresh  = thresh;
physio.verbose = verbose;
physio.ons_secs = ons_secs;

%% call functions for specific initial value settings (e.g. 3T Philips system)
switch default_scheme
    case 'Philips'
        physio = tapas_physio_init_philips(physio);
end